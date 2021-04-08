#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

struct LearningSolver
    py::PyCall.PyObject
end

function LearningSolver(;
                        optimizer,
                        kwargs...,
                       )::LearningSolver
    py = miplearn.LearningSolver(
        ;
        kwargs...,
        solver=JuMPSolver(optimizer=optimizer),
    )
    return LearningSolver(py)
end

solve!(solver::LearningSolver, instance; kwargs...) =
    solver.py.solve(instance; kwargs...)

fit!(solver::LearningSolver, instances; kwargs...) =
    solver.py.fit(instances; kwargs...)

export LearningSolver