using JuMP
import MIPLearn: from_str_array, to_str_array

function build_model()
    data = SetCoverData(
        costs = [5, 10, 12, 6, 8],
        incidence_matrix = [
            1 0 0 1 0
            1 1 0 0 0
            0 0 1 1 1
        ],
    )
    return build_setcover_model(data)
end

function test_solvers_jump()
    test_solvers_jump_extract()
    test_solvers_jump_add_constrs()
    test_solvers_jump_fix_vars()
    test_solvers_jump_warm_starts()
    test_solvers_jump_write()
end

function test_solvers_jump_extract()
    h5 = H5File(tempname(), "w")

    function test_scalar(key, expected)
        actual = h5.get_scalar(key)
        @test actual !== nothing
        @test actual == expected
    end

    function test_sparse(key, expected)
        actual = h5.get_sparse(key)
        @test actual !== nothing
        @test all(actual == expected)
    end

    function test_str_array(key, expected)
        actual = from_str_array(h5.get_array(key))
        @debug actual, expected
        @test actual !== nothing
        @test all(actual .== expected)
    end


    function test_array(key, expected)
        actual = h5.get_array(key)
        @debug actual, expected
        @test actual !== nothing
        @test all(actual .≈ expected)
    end

    model = build_model()
    model.extract_after_load(h5)
    test_sparse(
        "static_constr_lhs",
        [
            1 0 0 1 0
            1 1 0 0 0
            0 0 1 1 1
        ],
    )
    test_str_array("static_constr_names", ["eqs[0]", "eqs[1]", "eqs[2]"])
    test_array("static_constr_rhs", [1, 1, 1])
    test_str_array("static_constr_sense", [">", ">", ">"])
    test_scalar("static_obj_offset", 0)
    test_scalar("static_sense", "min")
    test_array("static_var_lower_bounds", [0, 0, 0, 0, 0])
    test_str_array("static_var_names", ["x[0]", "x[1]", "x[2]", "x[3]", "x[4]"])
    test_array("static_var_obj_coeffs", [5, 10, 12, 6, 8])
    test_str_array("static_var_types", ["B", "B", "B", "B", "B"])
    test_array("static_var_upper_bounds", [1, 1, 1, 1, 1])

    relaxed = model.relax()
    relaxed.optimize()
    relaxed.extract_after_lp(h5)
    test_array("lp_constr_dual_values", [0, 10, 6])
    test_array("lp_constr_slacks", [1, 0, 0])
    test_scalar("lp_obj_value", 11)
    test_array("lp_var_reduced_costs", [-5, 0, 6, 0, 2])
    test_array("lp_var_values", [1, 0, 0, 1, 0])
    test_str_array("lp_var_basis_status", ["U", "B", "L", "B", "L"])
    test_str_array("lp_constr_basis_status", ["B", "N", "N"])
    test_array("lp_constr_sa_rhs_up", [2, 2, 1])
    test_array("lp_constr_sa_rhs_down", [-Inf, 1, 0])
    test_array("lp_var_sa_obj_up", [10, Inf, Inf, 8, Inf])
    test_array("lp_var_sa_obj_down", [-Inf, 5, 6, 0, 6])
    test_array("lp_var_sa_ub_up", [1, Inf, Inf, Inf, Inf])
    test_array("lp_var_sa_ub_down", [0, 0, 0, 1, 0])
    test_array("lp_var_sa_lb_up", [1, 0, 1, 1, 1])
    test_array("lp_var_sa_lb_down", [-Inf, -Inf, 0, -Inf, 0])
    lp_wallclock_time = h5.get_scalar("lp_wallclock_time")
    @test lp_wallclock_time >= 0

    model.optimize()
    model.extract_after_mip(h5)
    test_array("mip_constr_slacks", [1, 0, 0])
    test_array("mip_var_values", [1.0, 0.0, 0.0, 1.0, 0.0])
    test_scalar("mip_gap", 0)
    test_scalar("mip_obj_bound", 11.0)
    test_scalar("mip_obj_value", 11.0)
    mip_wallclock_time = h5.get_scalar("mip_wallclock_time")
    @test mip_wallclock_time >= 0
end

function test_solvers_jump_add_constrs()
    h5 = H5File(tempname(), "w")
    model = build_model()
    model.extract_after_load(h5)
    model.add_constrs(
        to_str_array(["x[2]", "x[3]"]),
        [
            0 1
            1 0
        ],
        to_str_array(["=", "="]),
        [0, 0],
    )
    model.optimize()
    model.extract_after_mip(h5)
    @test all(h5.get_array("mip_var_values") .≈ [1, 0, 0, 0, 1])
end

function test_solvers_jump_fix_vars()
    h5 = H5File(tempname(), "w")
    model = build_model()
    model.extract_after_load(h5)
    model.fix_variables(
        to_str_array(["x[2]", "x[3]"]),
        [0, 0],
    )
    model.optimize()
    model.extract_after_mip(h5)
    @test all(h5.get_array("mip_var_values") .≈ [1, 0, 0, 0, 1])
end

function test_solvers_jump_warm_starts()
    # TODO: Check presence of warm start on log file
    h5 = H5File(tempname(), "w")
    model = build_model()
    model.extract_after_load(h5)
    model.set_warm_starts(
        to_str_array(["x[0]", "x[1]", "x[2]", "x[3]", "x[4]"]),
        [1 0 0 0 1],
    )
    model.optimize()
end

function test_solvers_jump_write()
    mps_filename = "$(tempname()).mps"
    model = build_model()
    model.write(mps_filename)
    @test isfile(mps_filename)
    rm(mps_filename)
end