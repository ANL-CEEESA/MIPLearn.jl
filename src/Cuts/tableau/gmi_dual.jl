#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Printf
using JuMP
using HiGHS
using Random
using DataStructures

global ExpertDualGmiComponent = PyNULL()
global KnnDualGmiComponent = PyNULL()

Base.@kwdef mutable struct _KnnDualGmiData
    k = nothing
    extractor = nothing
    train_h5 = nothing
    model = nothing
    strategy = nothing
end

function collect_gmi_dual(
    mps_filename;
    optimizer,
    max_rounds = 10,
    max_cuts_per_round = 500,
)
    reset_timer!()

    @timeit "Read H5" begin
        h5_filename = replace(mps_filename, ".mps.gz" => ".h5")
        h5 = H5File(h5_filename)
        sol_opt_dict = Dict(
            zip(
                h5.get_array("static_var_names"),
                convert(Array{Float64}, h5.get_array("mip_var_values")),
            ),
        )
        obj_mip = h5.get_scalar("mip_obj_value")
        h5.file.close()
    end

    # Define relative MIP gap
    gap(v) = 100 * abs(obj_mip - v) / abs(obj_mip)

    @timeit "Initialize" begin
        stats_obj = []
        stats_gap = []
        stats_ncuts = []
        original_basis = nothing
        all_cuts = nothing
        all_cuts_bases = nothing
        all_cuts_rows = nothing
        last_round_obj = nothing
    end

    @timeit "Read problem" begin
        model = read_from_file(mps_filename)
        set_optimizer(model, optimizer)
        obj_original = objective_function(model)
    end

    for round = 1:max_rounds
        @info "Round $(round)..."

        @timeit "Convert model to standard form" begin
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

        @timeit "Optimize standard model" begin
            @info "Optimizing standard model..."
            optimize!(model_s)
            obj = objective_value(model_s)
            if round == 1
                push!(stats_obj, obj)
                push!(stats_gap, gap(obj))
                push!(stats_ncuts, 0)
            else
                if obj ≈ last_round_obj
                    @info ("No improvement in obj value. Aborting.")
                    break
                end
            end
            if termination_status(model_s) != MOI.OPTIMAL
                error("Non-optimal termination status")
            end
            last_round_obj = obj
        end

        @timeit "Select tableau rows" begin
            basis = get_basis(model_s)
            if round == 1
                original_basis = basis
            end
            sol_frac = get_x(model_s)
            selected_rows =
                select_gmi_rows(data_s, basis, sol_frac, max_rows = max_cuts_per_round)
        end

        @timeit "Compute tableau rows" begin
            tableau = compute_tableau(data_s, basis, x = sol_frac, rows = selected_rows)

            # Assert tableau rows have been computed correctly
            assert_eq(tableau.lhs * sol_frac, tableau.rhs, atol=1e-3)
            assert_eq(tableau.lhs * sol_opt_s, tableau.rhs, atol=1e-3)
        end

        @timeit "Compute GMI cuts" begin
            cuts_s = compute_gmi(data_s, tableau)

            # Assert cuts have been generated correctly
            assert_cuts_off(cuts_s, sol_frac)
            assert_does_not_cut_off(cuts_s, sol_opt_s)

            # Abort if no cuts are left
            if length(cuts_s.lb) == 0
                @info "No cuts generated. Aborting."
                break
            else
                @info "Generated $(length(cuts_s.lb)) cuts"
            end
        end

        @timeit "Add GMI cuts to original model" begin
            @timeit "Convert to original form" begin
                cuts = backwards(transforms, cuts_s)
            end

            @timeit "Prepare bv" begin
                bv = repeat([basis], length(selected_rows))
            end

            @timeit "Append matrices" begin
                if round == 1
                    all_cuts = cuts
                    all_cuts_bases = bv
                    all_cuts_rows = selected_rows
                else
                    all_cuts.lhs = [all_cuts.lhs; cuts.lhs]
                    all_cuts.lb = [all_cuts.lb; cuts.lb]
                    all_cuts.ub = [all_cuts.ub; cuts.ub]
                    all_cuts_bases = [all_cuts_bases; bv]
                    all_cuts_rows = [all_cuts_rows; selected_rows]
                end
            end

            @timeit "Add to model" begin
                @info "Adding $(length(all_cuts.lb)) constraints to original model"
                constrs, gmi_exps = add_constraint_set_dual_v2(model, all_cuts)
            end
        end

        @timeit "Optimize original model" begin
            set_objective_function(model, obj_original)
            undo_relax = relax_integrality(model)
            @info "Optimizing original model (constr)..."
            optimize!(model)
            obj = objective_value(model)
            push!(stats_obj, obj)
            push!(stats_gap, gap(obj))
            sp = [shadow_price(c) for c in constrs]
            undo_relax()
            useful = [abs(sp[i]) > 1e-6 for (i, _) in enumerate(constrs)]
            keep = findall(useful .== true)
        end

        @timeit "Filter out useless cuts" begin
            @info "Keeping $(length(keep)) useful cuts"
            all_cuts.lhs = all_cuts.lhs[keep, :]
            all_cuts.lb = all_cuts.lb[keep]
            all_cuts.ub = all_cuts.ub[keep]
            all_cuts_bases = all_cuts_bases[keep, :]
            all_cuts_rows = all_cuts_rows[keep, :]
            push!(stats_ncuts, length(all_cuts_rows))
            if isempty(keep)
                break
            end
        end

        @timeit "Update obj function of original model" begin
            delete.(model, constrs)
            set_objective_function(
                model,
                obj_original -
                sum(sp[i] * gmi_exps[i] for (i, c) in enumerate(constrs) if useful[i]),
            )
        end
    end

    @timeit "Store cuts in H5 file" begin
        if all_cuts !== nothing
            ncuts = length(all_cuts_rows)
            total =
                length(original_basis.var_basic) +
                length(original_basis.var_nonbasic) +
                length(original_basis.constr_basic) +
                length(original_basis.constr_nonbasic)
            all_cuts_basis_sizes = Array{Int64,2}(undef, ncuts, 4)
            all_cuts_basis_vars = Array{Int64,2}(undef, ncuts, total)
            for i = 1:ncuts
                vb = all_cuts_bases[i].var_basic
                vn = all_cuts_bases[i].var_nonbasic
                cb = all_cuts_bases[i].constr_basic
                cn = all_cuts_bases[i].constr_nonbasic
                all_cuts_basis_sizes[i, :] = [length(vb) length(vn) length(cb) length(cn)]
                all_cuts_basis_vars[i, :] = [vb' vn' cb' cn']
            end
            @info "Storing $(length(all_cuts.ub)) GMI cuts..."
            h5 = H5File(h5_filename)
            h5.put_sparse("cuts_lhs", all_cuts.lhs)
            h5.put_array("cuts_lb", all_cuts.lb)
            h5.put_array("cuts_ub", all_cuts.ub)
            h5.put_array("cuts_basis_vars", all_cuts_basis_vars)
            h5.put_array("cuts_basis_sizes", all_cuts_basis_sizes)
            h5.put_array("cuts_rows", all_cuts_rows)
            h5.file.close()
        end
    end

    to = TimerOutputs.get_defaulttimer()
    stats_time = TimerOutputs.tottime(to) / 1e9
    print_timer()

    return OrderedDict(
        "instance" => mps_filename,
        "max_rounds" => max_rounds,
        "rounds" => length(stats_obj) - 1,
        "obj_mip" => obj_mip,
        "stats_obj" => stats_obj,
        "stats_gap" => stats_gap,
        "stats_ncuts" => stats_ncuts,
        "stats_time" => stats_time,
    )
end

function ExpertDualGmiComponent_before_mip(test_h5, model, _)
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
    all_cuts = nothing

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

        if all_cuts === nothing
            all_cuts = cuts
        else
            all_cuts.lhs = [all_cuts.lhs; cuts.lhs]
            all_cuts.lb = [all_cuts.lb; cuts.lb]
            all_cuts.ub = [all_cuts.ub; cuts.ub]
        end
    end

    # Strategy 1: Add all cuts during the first call
    function cut_callback_1(cb_data)
        if all_cuts !== nothing
            constrs = build_constraints(model, all_cuts)
            @info "Enforcing $(length(constrs)) cuts..."
            for c in constrs
                MOI.submit(model, MOI.UserCut(cb_data), c)
            end
            all_cuts = nothing
        end
    end

    # Strategy 2: Add violated cuts repeatedly until unable to separate
    callback_disabled = false
    function cut_callback_2(cb_data)
        if callback_disabled
            return
        end
        x = all_variables(model)
        x_val = callback_value.(cb_data, x)
        lhs_val = all_cuts.lhs * x_val
        is_violated = lhs_val .> all_cuts.ub
        selected_idx = findall(is_violated .== true)
        selected_cuts = ConstraintSet(
            lhs=all_cuts.lhs[selected_idx, :],
            ub=all_cuts.ub[selected_idx],
            lb=all_cuts.lb[selected_idx],
        )
        constrs = build_constraints(model, selected_cuts)
        if length(constrs) > 0
            @info "Enforcing $(length(constrs)) cuts..."
            for c in constrs
                MOI.submit(model, MOI.UserCut(cb_data), c)
            end
        else
            @info "No violated cuts found. Disabling callback."
            callback_disabled = true
        end
    end

    # Set up cut callback
    set_attribute(model, MOI.UserCutCallback(), cut_callback_1)
    # set_attribute(model, MOI.UserCutCallback(), cut_callback_2)

    stats = Dict()
    stats["ExpertDualGmi: cuts"] = length(all_cuts.lb)
    stats["ExpertDualGmi: time convert"] = stats_time_convert
    stats["ExpertDualGmi: time tableau"] = stats_time_tableau
    stats["ExpertDualGmi: time gmi"] = stats_time_gmi
    return stats
end

function add_constraint_set_dual_v2(model::JuMP.Model, cs::ConstraintSet)
    vars = all_variables(model)
    nrows, ncols = size(cs.lhs)

    @timeit "Transpose LHS" begin
        lhs_t = spzeros(ncols, nrows)
        ftranspose!(lhs_t, cs.lhs, x -> x)
        lhs_t_rows = rowvals(lhs_t)
        lhs_t_vals = nonzeros(lhs_t)
    end

    constrs = []
    gmi_exps = []
    for i = 1:nrows
        c = nothing
        gmi_exp = nothing
        gmi_exp2 = nothing
        @timeit "Build expr" begin
            expr = AffExpr()
            for k in nzrange(lhs_t, i)
                add_to_expression!(expr, lhs_t_vals[k], vars[lhs_t_rows[k]])
            end
        end
        @timeit "Add constraints" begin
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
        end
        @timeit "Update structs" begin
            push!(constrs, c)
            push!(gmi_exps, gmi_exp)
            if !isnothing(gmi_exp2)
                push!(gmi_exps, gmi_exp2)
            end
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
    @timeit "Read problem data" begin
        data = ProblemData(model)
    end
    @timeit "Convert to standard form" begin
        data_s, transforms = convert_to_standard_form(data)
    end

    @timeit "Collect cuts from H5 files" begin
        cut_basis_vars = nothing
        cut_basis_sizes = nothing
        cut_rows = nothing
        for h5_filename in train_h5
            h5 = H5File(h5_filename)
            cut_basis_vars_sample = h5.get_array("cuts_basis_vars")
            cut_basis_sizes_sample = h5.get_array("cuts_basis_sizes")
            cut_rows_sample = h5.get_array("cuts_rows")
            if cut_basis_vars === nothing
                cut_basis_vars = cut_basis_vars_sample
                cut_basis_sizes = cut_basis_sizes_sample
                cut_rows = cut_rows_sample
            else
                cut_basis_vars = [cut_basis_vars; cut_basis_vars_sample]
                cut_basis_sizes = [cut_basis_sizes; cut_basis_sizes_sample]
                cut_rows = [cut_rows; cut_rows_sample]
            end
            h5.close()
        end
        ncuts, nvars = size(cut_basis_vars)
    end

    @timeit "Group cuts by tableau basis" begin
        vars_to_unique_basis_offset = Dict()
        unique_basis_vars = Matrix{Int}(undef, 0, nvars)
        unique_basis_sizes = Matrix{Int}(undef, 0, 4)
        unique_basis_rows = Dict{Int,Set{Int}}()
        for i in 1:ncuts
            vars = cut_basis_vars[i, :]
            sizes = cut_basis_sizes[i, :]
            row = cut_rows[i]
            if vars ∉ keys(vars_to_unique_basis_offset)
                offset = size(unique_basis_vars)[1] + 1
                vars_to_unique_basis_offset[vars] = offset
                unique_basis_vars = [unique_basis_vars; vars']
                unique_basis_sizes = [unique_basis_sizes; sizes']
                unique_basis_rows[offset] = Set()
            end
            offset = vars_to_unique_basis_offset[vars]
            push!(unique_basis_rows[offset], row)
        end
    end

    @timeit "Compute tableaus and cuts" begin
        all_cuts = nothing
        for (offset, rows) in unique_basis_rows
            vbb, vnn, cbb, cnn = unique_basis_sizes[offset, :]
            current_basis = Basis(;
                var_basic = unique_basis_vars[offset, 1:vbb],
                var_nonbasic = unique_basis_vars[offset, vbb+1:vbb+vnn],
                constr_basic = unique_basis_vars[offset, vbb+vnn+1:vbb+vnn+cbb],
                constr_nonbasic = unique_basis_vars[offset, vbb+vnn+cbb+1:vbb+vnn+cbb+cnn],
            )

            tableau = compute_tableau(data_s, current_basis; rows=collect(rows))
            cuts_s = compute_gmi(data_s, tableau)
            cuts = backwards(transforms, cuts_s)

            if all_cuts === nothing
                all_cuts = cuts
            else
                all_cuts.lhs = [all_cuts.lhs; cuts.lhs]
                all_cuts.lb = [all_cuts.lb; cuts.lb]
                all_cuts.ub = [all_cuts.ub; cuts.ub]
            end
        end
    end
    return all_cuts
end

function _dualgmi_set_callback(model, all_cuts)
    function cut_callback(cb_data)
        if all_cuts !== nothing
            constrs = build_constraints(model, all_cuts)
            @info "Enforcing $(length(constrs)) cuts..."
            for c in constrs
                MOI.submit(model, MOI.UserCut(cb_data), c)
            end
            all_cuts = nothing
        end
    end
    set_attribute(model, MOI.UserCutCallback(), cut_callback)
end

function KnnDualGmiComponent_fit(data::_KnnDualGmiData, train_h5)
    x = hcat([_dualgmi_features(filename, data.extractor) for filename in train_h5]...)'
    model = pyimport("sklearn.neighbors").NearestNeighbors(n_neighbors = length(train_h5))
    model.fit(x)
    data.model = model
    data.train_h5 = train_h5
end


function KnnDualGmiComponent_before_mip(data::_KnnDualGmiData, test_h5, model, _)
    reset_timer!()

    @timeit "Extract features" begin
        x = _dualgmi_features(test_h5, data.extractor)
        x = reshape(x, 1, length(x))
    end

    @timeit "Find neighbors" begin
        neigh_dist, neigh_ind = data.model.kneighbors(x, return_distance = true)
        neigh_ind = neigh_ind .+ 1
        N = length(neigh_dist)

        if data.strategy == "near"
            selected = collect(1:(data.k))
        elseif data.strategy == "far"
            selected = collect((N - data.k + 1) : N)
        elseif data.strategy == "rand"
            selected = shuffle(collect(1:N))[1:(data.k)]
        else
            error("unknown strategy: $(data.strategy)")
        end

        @info "Dual GMI: Selected neighbors ($(data.strategy)):"
        neigh_dist = neigh_dist[selected]
        neigh_ind = neigh_ind[selected]
        for i in 1:data.k
            h5_filename = data.train_h5[neigh_ind[i]]
            dist = neigh_dist[i]
            @info "    $(h5_filename) dist=$(dist)"
        end
    end

    @info "Dual GMI: Generating cuts..."
    @timeit "Generate cuts" begin
        time_generate = @elapsed begin
            cuts = _dualgmi_generate(data.train_h5[neigh_ind], model)
        end
        @info "Dual GMI: Generated $(length(cuts.lb)) unique cuts in $(time_generate) seconds"
    end

    @timeit "Set callback" begin
        _dualgmi_set_callback(model, cuts)
    end

    print_timer()

    stats = Dict()
    stats["KnnDualGmi: k"] = data.k
    stats["KnnDualGmi: strategy"] = data.strategy
    stats["KnnDualGmi: cuts"] = length(cuts.lb)
    stats["KnnDualGmi: time generate"] = time_generate
    return stats
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
        function __init__(self; extractor, k = 3, strategy = "near")
            self.data = _KnnDualGmiData(; extractor, k, strategy)
        end
        function fit(self, train_h5)
            KnnDualGmiComponent_fit(self.data, train_h5)
        end
        function before_mip(self, test_h5, model, stats)
            return @time KnnDualGmiComponent_before_mip(self.data, test_h5, model.inner, stats)
        end
    end
    copy!(KnnDualGmiComponent, Class2)
end

export collect_gmi_dual, expert_gmi_dual, ExpertDualGmiComponent, KnnDualGmiComponent

