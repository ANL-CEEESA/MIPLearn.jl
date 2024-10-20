#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using HiGHS

global SetCoverData = PyNULL()
global SetCoverGenerator = PyNULL()

function __init_problems_setcover__()
    copy!(SetCoverData, pyimport("miplearn.problems.setcover").SetCoverData)
    copy!(SetCoverGenerator, pyimport("miplearn.problems.setcover").SetCoverGenerator)
end

function build_setcover_model_jump(data::Any; optimizer = HiGHS.Optimizer)
    if data isa String
        data = read_pkl_gz(data)
    end
    model = Model(optimizer)
    n_elements, n_sets = size(data.incidence_matrix)
    E = 0:n_elements-1
    S = 0:n_sets-1
    @variable(model, x[S], Bin)
    @objective(model, Min, sum(data.costs .* x))
    @constraint(
        model,
        eqs[e in E],
        sum(data.incidence_matrix[e+1, s+1] * x[s] for s in S) >= 1
    )
    return JumpModel(model)
end

export SetCoverData, SetCoverGenerator, build_setcover_model_jump
