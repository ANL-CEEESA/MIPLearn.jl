#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Random

"""
    StrongBranching(look_ahead::Int, max_calls::Int)

Branching strategy that selects a subset of fractional variables
as candidates (according to pseudocosts) the solves two linear
programming problems for each candidate.


"""
Base.@kwdef struct StrongBranching <: VariableBranchingRule
    look_ahead::Int = 10
    max_calls::Int = 100
    side_effect::Bool = true
end

function find_branching_var(rule::StrongBranching, node::Node, pool::NodePool)::Variable
    nfrac = length(node.fractional_variables)
    pseudocost_scores = [
        _pseudocost_score(
            node,
            pool,
            node.fractional_variables[j],
            node.fractional_values[j],
        ) for j = 1:nfrac
    ]
    σ = sortperm(pseudocost_scores, rev = true)
    sorted_vars = node.fractional_variables[σ]
    _strong_branch_start(node)
    no_improv_count, call_count = 0, 0
    max_score, max_var = pseudocost_scores[σ[1]], sorted_vars[1]
    for (i, var) in enumerate(sorted_vars)
        call_count += 1
        score = _strong_branch_score(
            node = node,
            pool = pool,
            var = var,
            x = node.fractional_values[σ[i]],
            side_effect = rule.side_effect,
        )
        if score > max_score
            max_score, max_var = score, var
            no_improv_count = 0
        else
            no_improv_count += 1
        end
        no_improv_count <= rule.look_ahead || break
        call_count <= rule.max_calls || break
    end
    _strong_branch_end(node)
    return max_var
end

function _strong_branch_score(;
    node::Node,
    pool::NodePool,
    var::Variable,
    x::Float64,
    side_effect::Bool,
)::Tuple{Float64,Int}
    obj_up, obj_down = 0, 0
    try
        obj_up, obj_down = probe(node.mip, var)
    catch
        @warn "strong branch error" var = var
    end
    obj_change_up = obj_up - node.obj
    obj_change_down = obj_down - node.obj
    if side_effect
        _update_var_history(
            pool = pool,
            var = var,
            x = x,
            obj_change_down = obj_change_down,
            obj_change_up = obj_change_up,
        )
    end
    return (obj_change_up * obj_change_down, var.index)
end

function _strong_branch_start(node::Node)
    set_bounds!(node.mip, node.branch_variables, node.branch_values, node.branch_values)
end

function _strong_branch_end(node::Node)
    nbranch = length(node.branch_variables)
    set_bounds!(node.mip, node.branch_variables, zeros(nbranch), ones(nbranch))
end
