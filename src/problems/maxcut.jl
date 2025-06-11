#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2025, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP

global MaxCutData = PyNULL()
global MaxCutGenerator = PyNULL()

function __init_problems_maxcut__()
    copy!(MaxCutData, pyimport("miplearn.problems.maxcut").MaxCutData)
    copy!(MaxCutGenerator, pyimport("miplearn.problems.maxcut").MaxCutGenerator)
end

function build_maxcut_model_jump(data::Any; optimizer)
    if data isa String
        data = read_pkl_gz(data)
    end
    nodes = collect(data.graph.nodes())
    edges = collect(data.graph.edges())
    model = Model(optimizer)
    @variable(model, x[nodes], Bin)
    @objective(
        model,
        Min,
        sum(-data.weights[i] * x[e[1]] * (1 - x[e[2]]) for (i, e) in enumerate(edges))
    )
    return JumpModel(model)
end

export MaxCutData, MaxCutGenerator, build_maxcut_model_jump
