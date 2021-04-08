#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Test
using MIPLearn
using Gurobi


@testset "LearningSolver" begin
    for optimizer in [Gurobi.Optimizer]
        instance = KnapsackInstance(
            [23., 26., 20., 18.],
            [505., 352., 458., 220.],
            67.0,
        )
        solver = LearningSolver(
            optimizer=optimizer,
            mode="heuristic",
        )
        stats = solve!(solver, instance)
        fit!(solver, [instance])
        solve!(solver, instance)
    end
end
