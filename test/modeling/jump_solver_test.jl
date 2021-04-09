#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Test
using MIPLearn
using Cbc
using PyCall

miplearn_tests = pyimport("miplearn.solvers.tests")

@testset "JuMPSolver" begin
    model = MIPLearn.knapsack_model(
        [23., 26., 20., 18.],
        [505., 352., 458., 220.],
        67.0,
    )
    instance = JuMPInstance(model)
    solver = JuMPSolver(optimizer=Cbc.Optimizer)
    miplearn_tests.test_internal_solver(solver, instance, model)
end