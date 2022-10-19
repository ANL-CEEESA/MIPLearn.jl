#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

"""
    ReliabilityBranching

Branching strategy that uses pseudocosts if there are enough observations
to make an accurate prediction of strong branching scores, or runs the
actual strong branching routine if not enough observations are available.
"""
Base.@kwdef mutable struct ReliabilityBranching <: VariableBranchingRule
    min_samples::Int = 8
    max_sb_calls::Int = 100
    look_ahead::Int = 10
    n_sb_calls::Int = 0
    side_effect::Bool = true
    max_iterations::Int = 1_000_000
    aggregation::Symbol = :prod
end

function find_branching_var(
    rule::ReliabilityBranching,
    node::Node,
    pool::NodePool,
)::Variable
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
    no_improv_count, n_sb_calls = 0, 0
    max_score, max_var = pseudocost_scores[σ[1]], sorted_vars[1]
    for (i, var) in enumerate(sorted_vars)
        use_strong_branch = true
        if n_sb_calls >= rule.max_sb_calls
            use_strong_branch = false
        else
            if var in keys(pool.var_history)
                varhist = pool.var_history[var]
                hlength = min(length(varhist.obj_ratio_up), length(varhist.obj_ratio_down))
                if hlength >= rule.min_samples
                    use_strong_branch = false
                end
            end
        end
        if use_strong_branch
            n_sb_calls += 1
            rule.n_sb_calls += 1
            score = _strong_branch_score(
                node = node,
                pool = pool,
                var = var,
                x = node.fractional_values[σ[i]],
                side_effect = rule.side_effect,
                max_iterations = rule.max_iterations,
                aggregation = rule.aggregation,
            )
        else
            score = pseudocost_scores[σ[i]]
        end
        if score > max_score
            max_score, max_var = score, var
            no_improv_count = 0
        else
            no_improv_count += 1
        end
        no_improv_count <= rule.look_ahead || break
    end
    _unset_node_bounds(node)
    return max_var
end
