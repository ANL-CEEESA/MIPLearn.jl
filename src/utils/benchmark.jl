#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using CSV
using DataFrames


mutable struct BenchmarkRunner
    solvers::Dict
    results::Union{Nothing,DataFrame}
    py::PyCall.PyObject

    function BenchmarkRunner(; solvers::Dict)
        return new(
            solvers,
            nothing,  # results
            miplearn.BenchmarkRunner(
                Dict(sname => solver.py for (sname, solver) in solvers),
            ),
        )
    end
end

function parallel_solve!(
    runner::BenchmarkRunner,
    instances::Vector{FileInstance};
    n_trials::Int = 3,
)::Nothing
    instances = repeat(instances, n_trials)
    for (solver_name, solver) in runner.solvers
        @info "benchmark $solver_name"
        stats = parallel_solve!(solver, instances, discard_output = true)
        for (i, s) in enumerate(stats)
            s["Solver"] = solver_name
            s["Instance"] = instances[i].filename
            s = Dict(k => isnothing(v) ? missing : v for (k, v) in s)
            if runner.results === nothing
                runner.results = DataFrame(s)
            else
                push!(runner.results, s, cols = :union)
            end
        end
        @info "benchmark $solver_name [done]"
    end
end

function fit!(runner::BenchmarkRunner, instances::Vector{FileInstance})::Nothing
    @python_call runner.py.fit([instance.py for instance in instances])
end

function write_csv!(runner::BenchmarkRunner, filename::AbstractString)::Nothing
    CSV.write(filename, runner.results)
    return
end

export BenchmarkRunner, parallel_solve!, fit!, write_csv!
