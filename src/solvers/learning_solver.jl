#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Distributed
using JLD2


function LearningSolver(
    optimizer_factory;
    components = nothing,
    mode::AbstractString = "exact",
    simulate_perfect::Bool = false,
    solve_lp::Bool = true,
    extract_sa::Bool = true,
    extract_lhs::Bool = true,
)::LearningSolver
    return LearningSolver(
        miplearn.LearningSolver(
            solver = JuMPSolver(optimizer_factory),
            mode = mode,
            solve_lp = solve_lp,
            simulate_perfect = simulate_perfect,
            components = components,
            extract_lhs = extract_lhs,
            extract_sa = extract_sa,
        ),
        optimizer_factory,
    )
end


function solve!(
    solver::LearningSolver,
    instance::Instance;
    tee::Bool = false,
    discard_output::Bool = false,
)
    return @python_call solver.py.solve(
        instance.py,
        tee = tee,
        discard_output = discard_output,
    )
end

function fit!(solver::LearningSolver, instances::Vector{<:Instance})
    @python_call solver.py.fit([instance.py for instance in instances])
    return
end


function save(filename::AbstractString, solver::LearningSolver)
    internal_solver = solver.py.internal_solver
    internal_solver_prototype = solver.py.internal_solver_prototype
    solver.py.internal_solver = nothing
    solver.py.internal_solver_prototype = nothing
    solver_py_filename = tempname()
    miplearn.write_pickle_gz(solver.py, solver_py_filename)
    solver_py = read(solver_py_filename)
    solver.py.internal_solver = internal_solver
    solver.py.internal_solver_prototype = internal_solver_prototype
    jldsave(
        filename;
        miplearn_version = "0.2",
        solver_py = solver_py,
        optimizer_factory = solver.optimizer_factory,
    )
    return
end


function load_solver(filename::AbstractString)::LearningSolver
    jldopen(filename, "r") do file
        solve_py_filename = tempname()
        write(solve_py_filename, file["solver_py"])
        solver_py = miplearn.read_pickle_gz(solve_py_filename)
        internal_solver = JuMPSolver(file["optimizer_factory"])
        solver_py.internal_solver_prototype = internal_solver
        return LearningSolver(solver_py, file["optimizer_factory"])
    end
end


export Instance, LearningSolver, solve!, fit!, parallel_solve!, save, load_solver
