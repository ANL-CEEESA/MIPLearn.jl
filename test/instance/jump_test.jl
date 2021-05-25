#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

@testset "JuMPInstance" begin
    @testset "save and load" begin
        # Create basic model
        model = Model()
        @variable(model, x, Bin)
        @variable(model, y, Bin)
        @objective(model, Max, x + y)
        @feature(x, [1.0])
        @category(x, "cat1")
        @feature(model, [5.0])

        # Solve
        instance = JuMPInstance(model)
        solver = LearningSolver(Gurobi.Optimizer)
        stats = solve!(solver, instance)
        @test length(instance.py.samples) == 1

        # Save model to file
        filename = tempname()
        save(filename, instance)
        @test isfile(filename)

        # Read model from file
        loaded = load_jump_instance(filename)
        x = variable_by_name(loaded.model, "x")
        @test loaded.model.ext[:miplearn][:variable_features][x] == [1.0]
        @test loaded.model.ext[:miplearn][:variable_categories][x] == "cat1"
        @test loaded.model.ext[:miplearn][:instance_features] == [5.0]
        @test length(loaded.py.samples) == 1
    end
end