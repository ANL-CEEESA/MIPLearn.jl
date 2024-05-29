#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import ..H5File

using OrderedCollections
using SparseArrays
using Statistics
using TimerOutputs

function collect_gmi(
    mps_filename;
    optimizer,
    max_rounds = 10,
    max_cuts_per_round = 100,
    atol = 1e-4,
)
    @info mps_filename
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
    all_cuts = nothing

    # Read problem
    model = read_from_file(mps_filename)

    for round = 1:max_rounds
        @info "Round $(round)..."

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
            assert_eq(obj, obj_lp)
            push!(stats_obj, obj)
            push!(stats_gap, gap(obj))
            push!(stats_ncuts, 0)
        end
        if termination_status(model_s) != MOI.OPTIMAL
            return
        end

        # Select tableau rows
        basis = get_basis(model_s)
        sol_frac = get_x(model_s)
        stats_time_select += @elapsed begin
            selected_rows =
                select_gmi_rows(data_s, basis, sol_frac, max_rows = max_cuts_per_round)
        end

        # Compute selected tableau rows
        stats_time_tableau += @elapsed begin
            tableau = compute_tableau(data_s, basis, sol_frac, rows = selected_rows)

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
                @info "No cuts generated. Stopping."
                break
            end
        end

        # Add GMI cuts to original problem
        cuts = backwards(transforms, cuts_s)
        assert_does_not_cut_off(cuts, sol_opt)
        constrs = add_constraint_set(model, cuts)

        # Optimize original form
        set_optimizer(model, optimizer)
        undo_relax = relax_integrality(model)
        optimize!(model)
        obj = objective_value(model)
        push!(stats_obj, obj)
        push!(stats_gap, gap(obj))

        # Store useful cuts; drop useless ones from the problem
        useful = [abs(shadow_price(c)) > atol for c in constrs]
        drop = findall(useful .== false)
        keep = findall(useful .== true)
        delete.(model, constrs[drop])
        if all_cuts === nothing
            all_cuts = cuts
        else
            all_cuts.lhs = [all_cuts.lhs; cuts.lhs[keep, :]]
            all_cuts.lb = [all_cuts.lb; cuts.lb[keep]]
            all_cuts.lb = [all_cuts.lb; cuts.lb[keep]]
        end
        push!(stats_ncuts, length(all_cuts.lb))

        undo_relax()
    end

    # Store cuts
    if all_cuts !== nothing
        @info "Storing $(length(all_cuts.ub)) GMI cuts..."
        h5 = H5File(h5_filename)
        h5.put_sparse("cuts_lhs", all_cuts.lhs)
        h5.put_array("cuts_lb", all_cuts.lb)
        h5.put_array("cuts_ub", all_cuts.ub)
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
        "obj_mip" => obj_mip,
        "stats_obj" => stats_obj,
        "stats_gap" => stats_gap,
        "stats_ncuts" => stats_ncuts,
    )
end

function select_gmi_rows(data, basis, x; max_rows = 10, atol = 1e-4)
    candidate_rows = [
        r for r = 1:length(basis.var_basic) if (
            (data.var_types[basis.var_basic[r]] != 'C') &&
            (frac(x[basis.var_basic[r]]) > atol) &&
            (frac2(x[basis.var_basic[r]]) > atol)
        )
    ]
    candidate_vals = frac.(x[basis.var_basic[candidate_rows]])
    score = abs.(candidate_vals .- 0.5)
    perm = sortperm(score)
    return [candidate_rows[perm[i]] for i = 1:min(length(perm), max_rows)]
end

function compute_gmi(data::ProblemData, tableau::Tableau)::ConstraintSet
    nrows, ncols = size(tableau.lhs)
    ub = Float64[Inf for _ = 1:nrows]
    lb = Float64[0.9999 for _ = 1:nrows]
    tableau_I, tableau_J, tableau_V = findnz(tableau.lhs)
    lhs_I = Int[]
    lhs_J = Int[]
    lhs_V = Float64[]
    @timeit "Compute coefficients" begin
        for k = 1:nnz(tableau.lhs)
            i::Int = tableau_I[k]
            j::Int = tableau_J[k]
            v::Float64 = 0.0
            frac_alpha_j = frac(tableau_V[k])
            alpha_j = tableau_V[k]
            beta = frac(tableau.rhs[i])
            if data.var_types[j] == 'C'
                if alpha_j >= 0
                    v = alpha_j / beta
                else
                    v = -alpha_j / (1 - beta)
                end
            else
                if frac_alpha_j < beta
                    v = frac_alpha_j / beta
                else
                    v = (1 - frac_alpha_j) / (1 - beta)
                end
            end
            if abs(v) > 1e-8
                push!(lhs_I, i)
                push!(lhs_J, tableau_J[k])
                push!(lhs_V, v)
            end
        end
        lhs = sparse(lhs_I, lhs_J, lhs_V, nrows, ncols)
    end
    return ConstraintSet(; lhs, ub, lb)
end

export compute_gmi,
    frac, select_gmi_rows, assert_cuts_off, assert_does_not_cut_off, collect_gmi
