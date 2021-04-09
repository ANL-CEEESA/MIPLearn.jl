#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

struct LearningSolver
    py::PyCall.PyObject
end

function LearningSolver(
    ;
    optimizer,
)::LearningSolver
    py = miplearn.LearningSolver(solver=JuMPSolver(optimizer=optimizer))
    return LearningSolver(py)
end

function solve!(solver::LearningSolver, model::Model)
    instance = JuMPInstance(model)
    mip_stats = solver.py.solve(instance)
    push!(
        model.ext[:miplearn][:training_samples],
        instance.training_data[1].__dict__,
    )
    return mip_stats
end

function fit!(solver::LearningSolver, models::Array{Model})
    instances = [JuMPInstance(m) for m in models]
    solver.py.fit(instances)
end

export LearningSolver
