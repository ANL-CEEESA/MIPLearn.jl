#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using MIPLearn
using Cbc


@testset "FileInstance" begin
    @testset "solve" begin
        model = build_knapsack_model()
        instance = JuMPInstance(model)
        filename = tempname()
        save(filename, instance)

        file_instance = FileInstance(filename)
        solver = LearningSolver(Cbc.Optimizer)
        solve!(solver, file_instance)

        loaded = load_instance(filename)
        @test length(loaded.samples) == 1
    end    
end
