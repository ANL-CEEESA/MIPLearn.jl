#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

"""
    PseudocostBranching()

Branching strategy that uses historical changes in objective value to estimate
strong branching scores at lower computational cost.
"""
struct PseudocostBranching <: VariableBranchingRule end

function find_branching_var(rule::PseudocostBranching, node::Node, pool::NodePool)::Variable
    scores = [
        _pseudocost_score(
            node,
            pool,
            node.fractional_variables[j],
            node.fractional_values[j],
        ) for j = 1:length(node.fractional_variables)
    ]
    _, max_offset = findmax(scores)
    return node.fractional_variables[max_offset]
end

function _pseudocost_score(
    node::Node,
    pool::NodePool,
    var::Variable,
    x::Float64,
)::Tuple{Float64,Int}
    f_up = x - floor(x)
    f_down = ceil(x) - x
    pseudo_up = pool.history.avg_pseudocost_up * f_up
    pseudo_down = pool.history.avg_pseudocost_down * f_down
    if var in keys(pool.var_history)
        if isfinite(pool.var_history[var].pseudocost_up)
            pseudo_up = pool.var_history[var].pseudocost_up * f_up
        end
        if isfinite(pool.var_history[var].pseudocost_down)
            pseudo_down = pool.var_history[var].pseudocost_down * f_down
        end
    end
    return (pseudo_up * f_up * pseudo_down * f_down, var.index)
end
