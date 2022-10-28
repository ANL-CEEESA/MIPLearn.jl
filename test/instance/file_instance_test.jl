#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using MIPLearn
using Cbc

@testset "FileInstance" begin
    @testset "Solve (knapsack)" begin
        data = KnapsackData()
        basename = tempname()
        MIPLearn.save_data("$basename.jld2", data)
        instance = FileInstance("$basename.jld2", build_knapsack_model)
        solver = LearningSolver(Cbc.Optimizer)
        solve!(solver, instance)
        h5 = Hdf5Sample("$basename.h5")
        @test h5.get_scalar("mip_wallclock_time") > 0
    end

    @testset "Solve (danoint)" begin
        data = Dict("filename" => joinpath(@__DIR__, "../fixtures/danoint.mps.gz"))
        build_model(data) = read_from_file(data["filename"])
        basename = tempname()
        MIPLearn.save_data("$basename.jld2", data)
        instance = FileInstance("$basename.jld2", build_model)
        solver = LearningSolver(optimizer_with_attributes(Cbc.Optimizer, "seconds" => 1.0))
        solve!(solver, instance)
        h5 = Hdf5Sample("$basename.h5")
        @test h5.get_scalar("mip_wallclock_time") > 0
    end

    @testset "Save and load data" begin
        filename = tempname()
        data = KnapsackData(
            weights = [5.0, 5.0, 5.0],
            prices = [1.0, 1.0, 1.0],
            capacity = 3.0,
        )
        MIPLearn.save_data(filename, data)
        loaded = MIPLearn.load_data(filename)
        @test loaded.weights == [5.0, 5.0, 5.0]
        @test loaded.prices == [1.0, 1.0, 1.0]
        @test loaded.capacity == 3.0
    end
end
