#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using MIPLearn


@testset "JuMPInstance" begin
    @testset "Save and load" begin
        # Build instance and solve
        model = model = build_knapsack_model()
        instance = JuMPInstance(model)
        solver = LearningSolver(Gurobi.Optimizer)
        stats = solve!(solver, instance)
        @test length(instance.py.samples) == 1

        # Save model to file
        filename = tempname()
        save(filename, instance)
        @test isfile(filename)

        # Read model from file
        loaded = load_instance(filename)
        x = variable_by_name(loaded.model, "x")
        @test loaded.model.ext[:miplearn][:variable_features][x] == [1.0]
        @test loaded.model.ext[:miplearn][:variable_categories][x] == "cat1"
        @test loaded.model.ext[:miplearn][:instance_features] == [5.0]
        @test length(loaded.py.samples) == 1
    end
end