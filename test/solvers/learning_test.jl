#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using MIPLearn
using Gurobi

@testset "LearningSolver" begin
    @testset "Model with annotations" begin
        model = build_knapsack_model()
        solver = LearningSolver(Gurobi.Optimizer)
        instance = JuMPInstance(model)
        stats = solve!(solver, instance)
        @test stats["mip_lower_bound"] == 11.0
        @test length(instance.py.samples) == 1
        fit!(solver, [instance])
        solve!(solver, instance)
    end

    @testset "Model without annotations" begin
        model = build_knapsack_model()
        solver = LearningSolver(Gurobi.Optimizer)
        instance = JuMPInstance(model)
        stats = solve!(solver, instance)
        @test stats["mip_lower_bound"] == 11.0
    end

    @testset "Save and load" begin
        solver = LearningSolver(Gurobi.Optimizer)
        solver.py.components = "Placeholder"
        filename = tempname()
        save(filename, solver)
        @test isfile(filename)
        loaded = load_solver(filename)
        @test loaded.py.components == "Placeholder"
    end
end
