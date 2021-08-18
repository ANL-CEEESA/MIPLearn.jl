#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Cbc
using JuMP
using MIPLearn

@testset "LearningSolver" begin
    @testset "Model with annotations" begin
        model = build_knapsack_model()
        solver = LearningSolver(Cbc.Optimizer)
        instance = JuMPInstance(model)
        stats = solve!(solver, instance)
        @test stats["mip_lower_bound"] == 11.0
        @test length(instance.samples) == 1
        fit!(solver, [instance])
        solve!(solver, instance)
    end

    @testset "Model without annotations" begin
        model = build_knapsack_model()
        solver = LearningSolver(Cbc.Optimizer)
        instance = JuMPInstance(model)
        stats = solve!(solver, instance)
        @test stats["mip_lower_bound"] == 11.0
    end

    @testset "Save and load" begin
        solver = LearningSolver(Cbc.Optimizer)
        solver.py.components = "Placeholder"
        filename = tempname()
        save(filename, solver)
        @test isfile(filename)
        loaded = load_solver(filename)
        @test loaded.py.components == "Placeholder"
    end

    @testset "Discard output" begin
        instance = build_knapsack_file_instance()
        solver = LearningSolver(Cbc.Optimizer)
        solve!(solver, instance, discard_output = true)
        loaded = load_instance(instance.filename)
        @test length(loaded.samples) == 0
    end
end
