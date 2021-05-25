#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using CSV
using DataFrames
using Gurobi


@testset "BenchmarkRunner" begin
    # Configure benchmark suite
    benchmark = BenchmarkRunner(
        solvers=Dict(
            "Baseline" => LearningSolver(Gurobi.Optimizer, components=[]),
            "Proposed" => LearningSolver(Gurobi.Optimizer),
        ),
    )

    # Solve instances in parallel
    instances = [
        build_knapsack_file_instance(),
        build_knapsack_file_instance(),
    ]
    parallel_solve!(benchmark, instances)

    # Write CSV
    csv_filename = tempname()
    write_csv!(benchmark, csv_filename)
    @test isfile(csv_filename)
    csv = DataFrame(CSV.File(csv_filename))
end
