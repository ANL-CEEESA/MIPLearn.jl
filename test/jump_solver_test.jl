#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Test
using MIPLearn
using Gurobi
using PyCall

miplearn_tests = pyimport("miplearn.solvers.tests")

@testset "JuMPSolver" begin
    for optimizer in [Gurobi.Optimizer]
        instance = KnapsackInstance(
            [23., 26., 20., 18.],
            [505., 352., 458., 220.],
            67.0,
        )
        model = instance.to_model()
        solver = JuMPSolver(optimizer=optimizer)
        miplearn_tests.test_internal_solver(solver, instance, model)
    end
end