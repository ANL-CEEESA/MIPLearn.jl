#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2025, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using PyCall

function test_problems_maxcut()
    np = pyimport("numpy")
    random = pyimport("random")
    scipy_stats = pyimport("scipy.stats")
    randint = scipy_stats.randint
    uniform = scipy_stats.uniform

    # Set random seed
    random.seed(42)
    np.random.seed(42)

    # Build random instance
    data = MaxCutGenerator(
        n = randint(low = 10, high = 11),
        p = uniform(loc = 0.5, scale = 0.0),
        fix_graph = false,
    ).generate(
        1,
    )[1]

    # Build model
    model = build_maxcut_model_jump(data, optimizer = SCIP.Optimizer)

    # Check static features
    h5 = H5File(tempname(), "w")
    model.extract_after_load(h5)
    obj_linear = h5.get_array("static_var_obj_coeffs")
    obj_quad = h5.get_array("static_var_obj_coeffs_quad")
    @test obj_linear == [3.0, 1.0, 3.0, 1.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0]
    @test obj_quad == [
        0.0 0.0 -1.0 1.0 -1.0 0.0 0.0 0.0 -1.0 -1.0
        0.0 0.0 1.0 -1.0 0.0 -1.0 -1.0 0.0 0.0 1.0
        0.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.0 -1.0 -1.0
        0.0 0.0 0.0 0.0 0.0 -1.0 1.0 -1.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 -1.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    ]

    # Check optimal solution
    model.optimize()
    model.extract_after_mip(h5)
    @test h5.get_scalar("mip_obj_value") == -4
    h5.close()
end
