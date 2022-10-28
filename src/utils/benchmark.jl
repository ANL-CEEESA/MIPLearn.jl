#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using CSV
using DataFrames
using OrderedCollections

function run_benchmarks(;
    optimizer,
    train_instances::Vector{<:AbstractString},
    test_instances::Vector{<:AbstractString},
    build_model::Function,
    progress::Bool = false,
    output_filename::String,
)
    solvers = OrderedDict(
        "baseline" => LearningSolver(optimizer),
        "ml-exact" => LearningSolver(optimizer),
        "ml-heuristic" => LearningSolver(optimizer, mode = "heuristic"),
    )

    #solve!(solvers["baseline"], train_instances, build_model; progress)
    fit!(solvers["ml-exact"], train_instances, build_model)
    fit!(solvers["ml-heuristic"], train_instances, build_model)

    stats = OrderedDict()
    for (solver_name, solver) in solvers
        stats[solver_name] = solve!(solver, test_instances, build_model; progress)
    end

    results = nothing
    for (solver_name, solver_stats) in stats
        for (i, s) in enumerate(solver_stats)
            s["Solver"] = solver_name
            s["Instance"] = test_instances[i]
            s = Dict(k => isnothing(v) ? missing : v for (k, v) in s)
            if results === nothing
                results = DataFrame(s)
            else
                push!(results, s, cols = :union)
            end
        end
    end
    CSV.write(output_filename, results)

    # fig_filename = "$(tempname()).svg"
    # df = pyimport("pandas").read_csv(csv_filename)
    # miplearn.benchmark.plot(df, output=fig_filename)
    # open(fig_filename) do f
    #     display("image/svg+xml", read(f, String))
    # end
    return
end

function run_benchmarks(; solvers, instance_filenames, build_model, output_filename)
    runner = BenchmarkRunner(; solvers)
    instances = [FileInstance(f, build_model) for f in instance_filenames]
    solve!(runner, instances)
    write_csv!(runner, output_filename)
end

export BenchmarkRunner, solve!, fit!, write_csv!
