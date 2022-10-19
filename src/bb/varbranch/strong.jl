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
    max_iterations::Int = 1_000_000
    aggregation::Symbol = :prod
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
    _set_node_bounds(node)
    no_improv_count, call_count = 0, 0
    max_score, max_var = (-Inf, -Inf), sorted_vars[1]
    for (i, var) in enumerate(sorted_vars)
        call_count += 1
        score = _strong_branch_score(
            node = node,
            pool = pool,
            var = var,
            x = node.fractional_values[σ[i]],
            side_effect = rule.side_effect,
            max_iterations = rule.max_iterations,
            aggregation = rule.aggregation,
        )
        # @show name(node.mip, var), round(score[1], digits=2)
        if score > max_score
            max_score, max_var = score, var
            no_improv_count = 0
        else
            no_improv_count += 1
        end
        no_improv_count <= rule.look_ahead || break
        call_count <= rule.max_calls || break
    end
    _unset_node_bounds(node)
    return max_var
end

function _strong_branch_score(;
    node::Node,
    pool::NodePool,
    var::Variable,
    x::Float64,
    side_effect::Bool,
    max_iterations::Int,
    aggregation::Symbol,
)::Tuple{Float64,Int}

    # Find current variable lower and upper bounds
    offset = findfirst(isequal(var), node.mip.int_vars)
    var_lb = node.mip.int_vars_lb[offset]
    var_ub = node.mip.int_vars_ub[offset]
    for (offset, v) in enumerate(node.branch_vars)
        if v == var
            var_lb = max(var_lb, node.branch_lb[offset])
            var_ub = min(var_ub, node.branch_ub[offset])
        end
    end

    obj_up, obj_down = 0, 0
    obj_up, obj_down = probe(node.mip, var, x, var_lb, var_ub, max_iterations)
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
    if aggregation == :prod
        return (obj_change_up * obj_change_down, var.index)
    elseif aggregation == :min
        sense = node.mip.sense
        return (min(sense * obj_up, sense * obj_down), var.index)
    else
        error("Unknown aggregation: $aggregation")
    end
end
