#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import ..Hdf5Sample

using OrderedCollections

function collect_gmi(
    mps_filename;
    optimizer,
    max_rounds=10,
    max_cuts_per_round=100,
)
    @info mps_filename
    reset_timer!()

    # Open HDF5 file
    h5_filename = replace(mps_filename, ".mps.gz" => ".h5")
    h5 = Hdf5Sample(h5_filename)

    # Read optimal solution
    sol_opt_dict = Dict(
        zip(
            h5.get_array("static_var_names"),
            convert(Array{Float64}, h5.get_array("mip_var_values")),
        )
    )

    # Read optimal value
    obj_mip = h5.get_scalar("mip_lower_bound")
    if obj_mip === nothing
        obj_mip = h5.get_scalar("mip_obj_value")
    end
    obj_lp = nothing
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

    for round in 1:max_rounds
        @info "Round $(round)..."
        
        stats_time_convert = @elapsed begin
            # Extract problem data
            data = ProblemData(model)

            # Construct optimal solution vector (with correct variable sequence)
            sol_opt = [sol_opt_dict[n] for n in data.var_names]

            # Assert optimal solution is feasible for the original problem
            @assert all(data.constr_lb .- 1e-3 .<= data.constr_lhs * sol_opt)
            @assert all(data.constr_lhs * sol_opt .<= data.constr_ub .+ 1e-3)

            # Convert to standard form
            data_s, transforms = convert_to_standard_form(data)
            model_s = to_model(data_s)
            set_optimizer(model_s, optimizer)
            relax_integrality(model_s)
    
            # Convert optimal solution to standard form
            sol_opt_s = forward(transforms, sol_opt)
    
            # Assert converted solution is feasible for standard form problem
            @assert data_s.constr_lhs * sol_opt_s ≈ data_s.constr_lb
        end

        # Optimize standard form
        optimize!(model_s)
        stats_time_solve += solve_time(model_s)
        obj = objective_value(model_s) + data_s.obj_offset
        if obj_lp === nothing
            obj_lp = obj
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
            selected_rows = select_gmi_rows(
                data_s,
                basis,
                sol_frac,
                max_rows=max_cuts_per_round,
            )
        end

        # Compute selected tableau rows
        stats_time_tableau += @elapsed begin
            tableau = compute_tableau(
                data_s,
                basis,
                sol_frac,
                rows=selected_rows,
            )

            # Assert tableau rows have been computed correctly
            @assert tableau.lhs * sol_frac ≈ tableau.rhs
            @assert tableau.lhs * sol_opt_s ≈ tableau.rhs
        end

        # Compute GMI cuts
        stats_time_gmi += @elapsed begin
            cuts_s = compute_gmi(data_s, tableau)

            # Assert cuts have been generated correctly
            try
                assert_cuts_off(cuts_s, sol_frac)
                assert_does_not_cut_off(cuts_s, sol_opt_s)
            catch
                @warn "Invalid cuts detected. Discarding round $round cuts and aborting."
                break
            end

            # Abort if no cuts are left
            if length(cuts_s.lb) == 0
                @info "No cuts generated. Aborting."
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
        useful = [
            abs(shadow_price(c)) > 1e-3
            for c in constrs
        ]
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
        h5 = Hdf5Sample(h5_filename)
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
        "obj_lp" => obj_lp,
        "stats_obj" => stats_obj,
        "stats_gap" => stats_gap,
        "stats_ncuts" => stats_ncuts,
    )
end

export collect_gmi