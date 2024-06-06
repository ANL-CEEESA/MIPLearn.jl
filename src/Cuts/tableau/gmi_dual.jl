#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Printf
using JuMP
using HiGHS

global ExpertDualGmiComponent = PyNULL()
global KnnDualGmiComponent = PyNULL()

Base.@kwdef mutable struct _KnnDualGmiData
    k = nothing
    extractor = nothing
    train_h5 = nothing
    model = nothing
end

Base.@kwdef mutable struct ConstraintSet_v2
    lhs::SparseMatrixCSC
    ub::Vector{Float64}
    lb::Vector{Float64}
    Bss::Vector{Basis}
    Bv::Vector{Int64}
end

function collect_gmi_dual(
    mps_filename;
    optimizer,
    max_rounds = 10,
    max_cuts_per_round = 500,
)
    reset_timer!()

    # Open HDF5 file
    h5_filename = replace(mps_filename, ".mps.gz" => ".h5")
    h5 = H5File(h5_filename)

    # Read optimal solution
    sol_opt_dict = Dict(
        zip(
            h5.get_array("static_var_names"),
            convert(Array{Float64}, h5.get_array("mip_var_values")),
        ),
    )

    # Read optimal value
    obj_mip = h5.get_scalar("mip_lower_bound")
    if obj_mip === nothing
        obj_mip = h5.get_scalar("mip_obj_value")
    end
    obj_lp = h5.get_scalar("lp_obj_value")
    h5.file.close()

    # Define relative MIP gap
    gap(v) = 100 * abs(obj_mip - v) / abs(v)

    # Initialize stats
    stats_obj = []
    stats_gap = []
    stats_ncuts = []
    stats_time_convert = 0
    stats_time_solve = 0
    stats_time_select = 0
    stats_time_tableau = 0
    stats_time_gmi = 0
    stats_time_dual = 0
    stats_time_dual_2 = 0
    all_cuts = nothing
    all_cuts_v2 = nothing
    cuts_all = nothing
    cuts_all_v2 = nothing
    original_basis = nothing

    # Read problem
    model = read_from_file(mps_filename)

    # Read original objective function
    or_obj_f = objective_function(model)
    revised_obj = objective_function(model)

    for round = 1:max_rounds
        @info "Round $(round)..."

        stats_time_convert = @elapsed begin
            # Update objective function
            set_objective_function(model, revised_obj)

            # Extract problem data
            data = ProblemData(model)

            # Construct optimal solution vector (with correct variable sequence)
            sol_opt = [sol_opt_dict[n] for n in data.var_names]

            # Assert optimal solution is feasible for the original problem
            assert_leq(data.constr_lb, data.constr_lhs * sol_opt)
            assert_leq(data.constr_lhs * sol_opt, data.constr_ub)

            # Convert to standard form
            data_s, transforms = convert_to_standard_form(data)
            model_s = to_model(data_s)
            set_optimizer(model_s, optimizer)
            relax_integrality(model_s)

            # Convert optimal solution to standard form
            sol_opt_s = forward(transforms, sol_opt)

            # Assert converted solution is feasible for standard form problem
            assert_eq(data_s.constr_lhs * sol_opt_s, data_s.constr_lb)

        end

        # Optimize standard form
        optimize!(model_s)
        stats_time_solve += solve_time(model_s)
        obj = objective_value(model_s) + data_s.obj_offset

        if round == 1
            # Assert standard form problem has same value as original
            if obj_lp !== nothing
                assert_eq(obj, obj_lp)
            end
            obj_lp = obj
            push!(stats_obj, obj)
            push!(stats_gap, gap(obj))
            push!(stats_ncuts, 0)
        end
        if termination_status(model_s) != MOI.OPTIMAL
            return
        end

        # Store original basis and select tableau rows
        basis = get_basis(model_s)
        if round == 1
            original_basis = basis
        end
        sol_frac = get_x(model_s)
        stats_time_select += @elapsed begin
            selected_rows =
                select_gmi_rows(data_s, basis, sol_frac, max_rows = max_cuts_per_round)
        end

        # Compute selected tableau rows
        stats_time_tableau += @elapsed begin
            tableau = compute_tableau(data_s, basis, x = sol_frac, rows = selected_rows)

            # Assert tableau rows have been computed correctly
            assert_eq(tableau.lhs * sol_frac, tableau.rhs)
            assert_eq(tableau.lhs * sol_opt_s, tableau.rhs)

        end

        # Compute GMI cuts
        stats_time_gmi += @elapsed begin
            cuts_s = compute_gmi(data_s, tableau)

            # Assert cuts have been generated correctly
            assert_cuts_off(cuts_s, sol_frac)
            assert_does_not_cut_off(cuts_s, sol_opt_s)

            # Abort if no cuts are left
            if length(cuts_s.lb) == 0
                @info "No cuts generated. Aborting."
                break
            end
        end

        # Add GMI cuts to original problem
        cuts = backwards(transforms, cuts_s)
        if round == 1
            cuts_all = cuts
            basis_vec = repeat([basis], length(selected_rows))
            cuts_all_v2 =
                ConstraintSet_v2(cuts.lhs, cuts.ub, cuts.lb, basis_vec, selected_rows)
        else
            # v1 struct
            cuts_all.lb = [cuts_all.lb; cuts.lb]
            cuts_all.ub = [cuts_all.ub; cuts.ub]
            cuts_all.lhs = [cuts_all.lhs; cuts.lhs]

            # v2 struct
            cuts_all_v2.lb = [cuts_all_v2.lb; cuts.lb]
            cuts_all_v2.ub = [cuts_all_v2.ub; cuts.ub]
            cuts_all_v2.lhs = [cuts_all_v2.lhs; cuts.lhs]
            cuts_all_v2.Bss = [cuts_all_v2.Bss; repeat([basis], length(selected_rows))]
            cuts_all_v2.Bv = [cuts_all_v2.Bv; selected_rows]
        end
        constrs, gmi_exps = add_constraint_set_dual_v2(model, cuts_all)

        # Optimize original form
        set_objective_function(model, or_obj_f)
        set_optimizer(model, optimizer)
        undo_relax = relax_integrality(model)
        optimize!(model)
        obj = objective_value(model)
        push!(stats_obj, obj)
        push!(stats_gap, gap(obj))

        # Reoptimize with updated obj function
        stats_time_dual += @elapsed begin
            revised_obj = (
                or_obj_f - sum(
                    shadow_price(c) * gmi_exps[iz] for (iz, c) in enumerate(constrs)
                )
            )
            delete.(model, constrs)
            set_objective_function(model, revised_obj)
            set_optimizer(model, optimizer)
            optimize!(model)
            n_obj = objective_value(model)
            assert_eq(obj, n_obj, atol = 0.01)
        end
        undo_relax()
    end

    # Filter out useless cuts
    stats_time_dual_2 += @elapsed begin
        set_objective_function(model, or_obj_f)
        keep = []
        obj_gmi = obj_lp
        if (cuts_all !== nothing)
            constrs, gmi_exps = add_constraint_set_dual_v2(model, cuts_all)
            for (i, c) in enumerate(constrs)
                set_name(c, @sprintf("gomory_%05d", i))
            end
            set_optimizer(model, optimizer)
            undo_relax = relax_integrality(model)
            optimize!(model)
            obj = objective_value(model)
            obj_gmi = obj
            push!(stats_obj, obj)
            push!(stats_gap, gap(obj))

            # Store useful cuts; drop useless ones from the problem
            useful = [-shadow_price(c) > 1e-3 for c in constrs]
            drop = findall(useful .== false)
            keep = findall(useful .== true)
            all_cuts = ConstraintSet(;
                lhs = cuts_all.lhs[keep, :],
                lb = cuts_all.lb[keep],
                ub = cuts_all.ub[keep],
            )
            all_cuts_v2 = ConstraintSet_v2(;
                lhs = cuts_all_v2.lhs[keep, :],
                lb = cuts_all_v2.lb[keep],
                ub = cuts_all_v2.ub[keep],
                Bss = cuts_all_v2.Bss[keep],
                Bv = cuts_all_v2.Bv[keep],
            )

            delete.(model, constrs[drop])
            undo_relax()
        end
    end
    basis = original_basis

    if all_cuts !== nothing
        cut_sizezz = length(all_cuts_v2.Bv)
        var_totall =
            length(basis.var_basic) +
            length(basis.var_nonbasic) +
            length(basis.constr_basic) +
            length(basis.constr_nonbasic)
        bm_size = Array{Int64,2}(undef, cut_sizezz, 4)
        basis_matrix = Array{Int64,2}(undef, cut_sizezz, var_totall)

        for ii = 1:cut_sizezz
            vb = all_cuts_v2.Bss[ii].var_basic
            vn = all_cuts_v2.Bss[ii].var_nonbasic
            cb = all_cuts_v2.Bss[ii].constr_basic
            cn = all_cuts_v2.Bss[ii].constr_nonbasic
            bm_size[ii, :] = [length(vb) length(vn) length(cb) length(cn)]
            basis_matrix[ii, :] = [vb' vn' cb' cn']
        end

        # Store cuts
        @info "Storing $(length(all_cuts.ub)) GMI cuts..."
        h5 = H5File(h5_filename)
        h5.put_sparse("cuts_lhs", all_cuts.lhs)
        h5.put_array("cuts_lb", all_cuts.lb)
        h5.put_array("cuts_ub", all_cuts.ub)
        h5.put_array("cuts_basis_vars", basis_matrix)
        h5.put_array("cuts_basis_sizes", bm_size)
        h5.put_array("cuts_rows", all_cuts_v2.Bv)
        h5.file.close()
    end

    return OrderedDict(
        "instance" => mps_filename,
        "max_rounds" => max_rounds,
        "rounds" => length(stats_obj) - 1,
        "time_convert" => stats_time_convert,
        "time_solve" => stats_time_solve,
        "time_tableau" => stats_time_tableau,
        "time_gmi" => stats_time_gmi,
        "time_dual" => stats_time_dual,
        "time_dual_2" => stats_time_dual_2,
        "obj_mip" => obj_mip,
        "obj_lp" => obj_lp,
        "stats_obj" => stats_obj,
        "stats_gap" => stats_gap,
        "stats_ncuts" => length(keep),
    )
end

function ExpertDualGmiComponent_before_mip(test_h5, model, stats)
    # Read cuts and optimal solution
    h5 = H5File(test_h5)
    sol_opt_dict = Dict(
        zip(
            h5.get_array("static_var_names"),
            convert(Array{Float64}, h5.get_array("mip_var_values")),
        ),
    )
    cut_basis_vars = h5.get_array("cuts_basis_vars")
    cut_basis_sizes = h5.get_array("cuts_basis_sizes")
    cut_rows = h5.get_array("cuts_rows")
    obj_mip = h5.get_scalar("mip_lower_bound")
    if obj_mip === nothing
        obj_mip = h5.get_scalar("mip_obj_value")
    end
    h5.close()

    # Initialize stats
    stats_time_convert = 0
    stats_time_tableau = 0
    stats_time_gmi = 0
    all_cuts = []

    stats_time_convert = @elapsed begin
        # Extract problem data
        data = ProblemData(model)

        # Construct optimal solution vector (with correct variable sequence)
        sol_opt = [sol_opt_dict[n] for n in data.var_names]

        # Assert optimal solution is feasible for the original problem
        assert_leq(data.constr_lb, data.constr_lhs * sol_opt)
        assert_leq(data.constr_lhs * sol_opt, data.constr_ub)

        # Convert to standard form
        data_s, transforms = convert_to_standard_form(data)
        model_s = to_model(data_s)
        set_optimizer(model_s, HiGHS.Optimizer)
        relax_integrality(model_s)

        # Convert optimal solution to standard form
        sol_opt_s = forward(transforms, sol_opt)

        # Assert converted solution is feasible for standard form problem
        assert_eq(data_s.constr_lhs * sol_opt_s, data_s.constr_lb)

    end

    current_basis = nothing
    for (r, row) in enumerate(cut_rows)
        stats_time_tableau += @elapsed begin
            if r == 1 || cut_basis_vars[r, :] != cut_basis_vars[r-1, :]
                vbb, vnn, cbb, cnn = cut_basis_sizes[r, :]
                current_basis = Basis(;
                    var_basic = cut_basis_vars[r, 1:vbb],
                    var_nonbasic = cut_basis_vars[r, vbb+1:vbb+vnn],
                    constr_basic = cut_basis_vars[r, vbb+vnn+1:vbb+vnn+cbb],
                    constr_nonbasic = cut_basis_vars[r, vbb+vnn+cbb+1:vbb+vnn+cbb+cnn],
                )
            end
            tableau = compute_tableau(data_s, current_basis, rows = [row])
            assert_eq(tableau.lhs * sol_opt_s, tableau.rhs)
        end
        stats_time_gmi += @elapsed begin
            cuts_s = compute_gmi(data_s, tableau)
            assert_does_not_cut_off(cuts_s, sol_opt_s)
        end
        cuts = backwards(transforms, cuts_s)
        assert_does_not_cut_off(cuts, sol_opt)
        push!(all_cuts, cuts)
    end

    function cut_callback(cb_data)
        if all_cuts !== nothing
            @info "Enforcing dual GMI cuts..."
            for cuts in all_cuts
                constrs = build_constraints(model, cuts)
                for c in constrs
                    MOI.submit(model, MOI.UserCut(cb_data), c)
                end
            end
            all_cuts = nothing
        end
    end

    # Set up cut callback
    set_attribute(model, MOI.UserCutCallback(), cut_callback)

    stats["gmi_time_convert"] = stats_time_convert
    stats["gmi_time_tableau"] = stats_time_tableau
    stats["gmi_time_gmi"] = stats_time_gmi
    return
end

function add_constraint_set_dual_v2(model::JuMP.Model, cs::ConstraintSet)
    vars = all_variables(model)
    nrows, ncols = size(cs.lhs)
    constrs = []
    gmi_exps = []
    for i = 1:nrows
        c = nothing
        gmi_exp = nothing
        gmi_exp2 = nothing
        expr = @expression(model, sum(cs.lhs[i, j] * vars[j] for j = 1:ncols))
        if isinf(cs.ub[i])
            c = @constraint(model, cs.lb[i] <= expr)
            gmi_exp = cs.lb[i] - expr
        elseif isinf(cs.lb[i])
            c = @constraint(model, expr <= cs.ub[i])
            gmi_exp = expr - cs.ub[i]
        else
            c = @constraint(model, cs.lb[i] <= expr <= cs.ub[i])
            gmi_exp = cs.lb[i] - expr
            gmi_exp2 = expr - cs.ub[i]
        end
        push!(constrs, c)
        push!(gmi_exps, gmi_exp)
        if !isnothing(gmi_exp2)
            push!(gmi_exps, gmi_exp2)
        end
    end
    return constrs, gmi_exps
end

function _dualgmi_features(h5_filename, extractor)
    h5 = H5File(h5_filename, "r")
    try
        return extractor.get_instance_features(h5)
    finally
        h5.close()
    end
end

function _dualgmi_generate(train_h5, model)
    data = ProblemData(model)
    data_s, transforms = convert_to_standard_form(data)
    all_cuts = []
    for h5_filename in train_h5
        h5 = H5File(h5_filename)
        cut_basis_vars = h5.get_array("cuts_basis_vars")
        cut_basis_sizes = h5.get_array("cuts_basis_sizes")
        cut_rows = h5.get_array("cuts_rows")
        h5.close()
        current_basis = nothing
        for (r, row) in enumerate(cut_rows)
            if r == 1 || cut_basis_vars[r, :] != cut_basis_vars[r-1, :]
                vbb, vnn, cbb, cnn = cut_basis_sizes[r, :]
                current_basis = Basis(;
                    var_basic = cut_basis_vars[r, 1:vbb],
                    var_nonbasic = cut_basis_vars[r, vbb+1:vbb+vnn],
                    constr_basic = cut_basis_vars[r, vbb+vnn+1:vbb+vnn+cbb],
                    constr_nonbasic = cut_basis_vars[r, vbb+vnn+cbb+1:vbb+vnn+cbb+cnn],
                )
            end
            tableau = compute_tableau(data_s, current_basis, rows = [row])
            cuts_s = compute_gmi(data_s, tableau)
            cuts = backwards(transforms, cuts_s)
            push!(all_cuts, cuts)
        end
    end
    return all_cuts
end

function _dualgmi_set_callback(model, all_cuts)
    function cut_callback(cb_data)
        if all_cuts !== nothing
            @info "Dual GMI: Submitting cuts..."
            for cuts in all_cuts
                constrs = build_constraints(model, cuts)
                for c in constrs
                    MOI.submit(model, MOI.UserCut(cb_data), c)
                end
            end
            all_cuts = nothing
        end
    end
    set_attribute(model, MOI.UserCutCallback(), cut_callback)
end

function KnnDualGmiComponent_fit(data::_KnnDualGmiData, train_h5)
    x = hcat([
        _dualgmi_features(filename, data.extractor)
        for filename in train_h5
    ]...)'
    model = pyimport("sklearn.neighbors").NearestNeighbors(n_neighbors=data.k)
    model.fit(x)
    data.model = model
    data.train_h5 = train_h5
end


function KnnDualGmiComponent_before_mip(data::_KnnDualGmiData, test_h5, model, stats)
    x = _dualgmi_features(test_h5, data.extractor)
    x = reshape(x, 1, length(x))
    selected = vec(data.model.kneighbors(x, return_distance=false)) .+ 1
    @info "Dual GMI: Nearest neighbors:"
    for h5_filename in data.train_h5[selected]
        @info "    $(h5_filename)"
    end
    cuts = _dualgmi_generate(data.train_h5[selected], model)
    _dualgmi_set_callback(model, cuts)
end

function __init_gmi_dual__()
    @pydef mutable struct Class1
        function fit(_, _) end
        function before_mip(self, test_h5, model, stats)
            ExpertDualGmiComponent_before_mip(test_h5, model.inner, stats)
        end
    end
    copy!(ExpertDualGmiComponent, Class1)

    @pydef mutable struct Class2
        function __init__(self; extractor, k = 3)
            self.data = _KnnDualGmiData(; extractor, k)
        end
        function fit(self, train_h5)
            KnnDualGmiComponent_fit(self.data, train_h5)
        end
        function before_mip(self, test_h5, model, stats)
            KnnDualGmiComponent_before_mip(self.data, test_h5, model.inner, stats)
        end
    end
    copy!(KnnDualGmiComponent, Class2)
end

export collect_gmi_dual, expert_gmi_dual, ExpertDualGmiComponent, KnnDualGmiComponent
