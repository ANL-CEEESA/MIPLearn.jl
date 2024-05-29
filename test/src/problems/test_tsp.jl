#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2024, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using GLPK
using JuMP

function test_problems_tsp()
    pdist = pyimport("scipy.spatial.distance").pdist
    squareform = pyimport("scipy.spatial.distance").squareform

    data = TravelingSalesmanData(
        n_cities = 6,
        distances = squareform(
            pdist([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 1.0], [3.0, 1.0]]),
        ),
    )
    model = build_tsp_model_jump(data, optimizer = GLPK.Optimizer)
    model.optimize()
    @test objective_value(model.inner) == 8.0
    return
end
