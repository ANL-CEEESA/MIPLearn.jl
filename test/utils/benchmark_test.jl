#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Cbc
using CSV
using DataFrames


@testset "BenchmarkRunner" begin
    @info "Building training data..."
    instances = [build_knapsack_file_instance(), build_knapsack_file_instance()]
    stats = parallel_solve!(LearningSolver(Cbc.Optimizer), instances)
    @test length(stats) == 2
    @test stats[1] !== nothing
    @test stats[2] !== nothing

    benchmark = BenchmarkRunner(
        solvers = Dict(
            "baseline" => LearningSolver(Cbc.Optimizer, components = []),
            "ml-exact" => LearningSolver(Cbc.Optimizer),
            "ml-heur" => LearningSolver(Cbc.Optimizer, mode = "heuristic"),
        ),
    )
    @info "Fitting..."
    fit!(benchmark, instances)

    @info "Benchmarking..."
    parallel_solve!(benchmark, instances, n_trials = 2)

    csv_filename = tempname()
    write_csv!(benchmark, csv_filename)
    @test isfile(csv_filename)
    csv = DataFrame(CSV.File(csv_filename))
    @test size(csv)[1] == 12
end
