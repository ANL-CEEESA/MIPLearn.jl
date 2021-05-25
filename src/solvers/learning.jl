#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Distributed


struct LearningSolver
    py::PyCall.PyObject
    optimizer_factory
end


function LearningSolver(optimizer_factory)::LearningSolver
    py = miplearn.LearningSolver(solver=JuMPSolver(optimizer_factory))
    return LearningSolver(
        py,
        optimizer_factory,
    )
end


function solve!(
    solver::LearningSolver,
    instance::Instance;
    tee::Bool = false,
)
    return @python_call solver.py.solve(instance.py, tee=tee)
end


function fit!(solver::LearningSolver, instances::Vector{<:Instance})
    @python_call solver.py.fit([instance.py for instance in instances])
end


function parallel_solve!(solver::LearningSolver, instances::Vector{FileInstance})
    filenames = [instance.filename for instance in instances]
    optimizer_factory = solver.optimizer_factory
    solver_filename = tempname()
    save(solver_filename, solver)
    @sync @distributed for filename in filenames
        s = load_solver(solver_filename)
        solve!(s, FileInstance(filename))
        nothing
    end
end


function save(filename::AbstractString, solver::LearningSolver)
    @info "Writing: $filename"
    time = @elapsed begin
        # Pickle solver.py
        internal_solver = solver.py.internal_solver
        internal_solver_prototype = solver.py.internal_solver_prototype
        solver.py.internal_solver = nothing
        solver.py.internal_solver_prototype = nothing
        solver_py_filename = tempname()
        miplearn.write_pickle_gz(solver.py, solver_py_filename, quiet=true)
        solver_py = read(solver_py_filename)
        solver.py.internal_solver = internal_solver
        solver.py.internal_solver_prototype = internal_solver_prototype

        jldsave(
            filename;
            miplearn_version="0.2",
            solver_py=solver_py,
            optimizer_factory=solver.optimizer_factory,
        )
    end
    @info @sprintf("File written in %.2f seconds", time)
    return
end


function load_solver(filename::AbstractString)::LearningSolver
    @info "Reading: $filename"
    solver = nothing
    time = @elapsed begin
        jldopen(filename, "r") do file
            _check_miplearn_version(file)
            solve_py_filename = tempname()
            write(solve_py_filename, file["solver_py"])
            solver_py = miplearn.read_pickle_gz(solve_py_filename, quiet=true)
            internal_solver = JuMPSolver(file["optimizer_factory"])
            solver_py.internal_solver_prototype = internal_solver
            solver = LearningSolver(
                solver_py,
                file["optimizer_factory"],
            )
        end
    end
    @info @sprintf("File read in %.2f seconds", time)
    return solver
end


export Instance,
       LearningSolver,
       solve!,
       fit!,
       parallel_solve!,
       save,
       load_solver
