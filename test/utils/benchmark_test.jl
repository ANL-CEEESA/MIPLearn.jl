#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using CSV
using DataFrames
using Gurobi


@testset "BenchmarkRunner" begin
    # Initialie instances and generate training data
    instances = [
        build_knapsack_file_instance(),
        build_knapsack_file_instance(),
    ]
    parallel_solve!(
        LearningSolver(Gurobi.Optimizer),
        instances,
    )

    # Fit and benchmark
    benchmark = BenchmarkRunner(
        solvers=Dict(
            "baseline" => LearningSolver(
                Gurobi.Optimizer,
                components=[],
            ),
            "ml-exact" => LearningSolver(
                Gurobi.Optimizer,
            ),
            "ml-heur" => LearningSolver(
                Gurobi.Optimizer,
                mode="heuristic",
            ),
        ),
    )
    fit!(benchmark, instances)
    parallel_solve!(benchmark, instances, n_trials=1)

    # Write CSV
    csv_filename = tempname()
    write_csv!(benchmark, csv_filename)
    @test isfile(csv_filename)
    csv = DataFrame(CSV.File(csv_filename))
    @test size(csv)[1] == 6
end
