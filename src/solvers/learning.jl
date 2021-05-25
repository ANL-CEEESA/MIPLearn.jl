#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

struct LearningSolver
    py::PyCall.PyObject
end


abstract type Instance
end


function LearningSolver(optimizer_factory)::LearningSolver
    py = miplearn.LearningSolver(solver=JuMPSolver(optimizer_factory))
    return LearningSolver(py)
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


export Instance,
       LearningSolver,
       solve!,
       fit!
