#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using CSV
using DataFrames


mutable struct BenchmarkRunner
    solvers::Dict
    results::Union{Nothing,DataFrame}
end


function BenchmarkRunner(; solvers::Dict)
    return BenchmarkRunner(
        solvers,
        nothing,  # results
    )
end


function parallel_solve!(
    runner::BenchmarkRunner,
    instances::Vector{FileInstance};
    n_trials::Int = 3,
)::Nothing
    for (solver_name, solver) in runner.solvers
        for i in 1:n_trials
            for instance in instances
                stats = solve!(solver, instance)
                stats["Solver"] = solver_name
                stats = Dict(k => isnothing(v) ? missing : v for (k, v) in stats)
                if runner.results === nothing
                    runner.results = DataFrame(stats)
                else
                    push!(runner.results, stats, cols=:union)
                end
            end
        end
    end
end


function fit!(
    runner::BenchmarkRunner,
    instances::Vector{FileInstance}
)::Nothing
    for (solver_name, solver) in runner.solvers
        fit!(solver, instances)
    end
end


function write_csv!(
    runner::BenchmarkRunner,
    filename::AbstractString,
)::Nothing
    @info "Writing: $filename"
    CSV.write(filename, runner.results)
    return
end


export BenchmarkRunner,
       parallel_solve!,
       fit!,
       write_csv!
