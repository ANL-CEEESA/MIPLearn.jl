#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2024, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using HiGHS

global MaxWeightStableSetData = PyNULL()
global MaxWeightStableSetGenerator = PyNULL()

function __init_problems_stab__()
    copy!(MaxWeightStableSetData, pyimport("miplearn.problems.stab").MaxWeightStableSetData)
    copy!(MaxWeightStableSetGenerator, pyimport("miplearn.problems.stab").MaxWeightStableSetGenerator)
end

function build_stab_model_jump(data::Any; optimizer=HiGHS.Optimizer)
    nx = pyimport("networkx")

    if data isa String
        data = read_pkl_gz(data)
    end
    model = Model(optimizer)

    # Variables and objective function
    nodes = data.graph.nodes
    x = @variable(model, x[nodes], Bin)
    @objective(model, Min, sum(-data.weights[i+1] * x[i] for i in nodes))

    # Edge inequalities
    for (i1, i2) in data.graph.edges
        @constraint(model, x[i1] + x[i2] <= 1, base_name = "eq_edge[$i1,$i2]")
    end

    function cuts_separate(cb_data)
        x_val = callback_value.(Ref(cb_data), x)
        violations = []
        for clique in nx.find_cliques(data.graph)
            if sum(x_val[i] for i in clique) > 1.0001
                push!(violations, sort(clique))
            end
        end
        return violations
    end

    function cuts_enforce(violations)
        @info "Adding $(length(violations)) clique cuts..."
        for clique in violations
            constr = @build_constraint(sum(x[i] for i in clique) <= 1)
            submit(model, constr)
        end
    end

    return JumpModel(
        model,
        cuts_separate=cuts_separate,
        cuts_enforce=cuts_enforce,
    )
end

export MaxWeightStableSetData, MaxWeightStableSetGenerator, build_stab_model_jump
