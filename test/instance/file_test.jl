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

        h5 = MIPLearn.Hdf5Sample(filename)
        @test h5.get_scalar("miplearn_version") == "0002"
        @test length(h5.get_bytes("mps")) > 0
        @test length(h5.get_scalar("jump_ext")) > 0

        file_instance = FileInstance(filename)
        solver = LearningSolver(Cbc.Optimizer)
        solve!(solver, file_instance)

        @test length(h5.get_vector("mip_var_values")) == 3
    end    
end
