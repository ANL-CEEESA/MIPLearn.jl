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
    @sync @distributed for filename in filenames
        s = LearningSolver(optimizer_factory)
        solve!(s, FileInstance(filename))
        nothing
    end
end


export Instance,
       LearningSolver,
       solve!,
       fit!,
       parallel_solve!
