#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import ..to_str_array

Base.@kwdef mutable struct ReliabilityBranchingStats
    branched_count::Vector{Int} = []
    num_strong_branch_calls = 0
    score_var_names::Vector{String} = []
    score_features::Vector{Vector{Float32}} = []
    score_targets::Vector{Float32} = []
end

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
    side_effect::Bool = true
    max_iterations::Int = 1_000_000
    aggregation::Symbol = :prod
    stats::ReliabilityBranchingStats = ReliabilityBranchingStats()
    collect::Bool = false
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

function find_branching_var(
    rule::ReliabilityBranching,
    node::Node,
    pool::NodePool,
)::Variable
    stats = rule.stats

    # Initialize statistics
    if isempty(stats.branched_count)
        stats.branched_count = zeros(node.mip.nvars)
    end

    # Sort variables by pseudocost score
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

    if rule.collect
        # Compute dynamic features for all fractional variables
        features = []
        for (i, var) in enumerate(sorted_vars)
            branched_count = stats.branched_count[var.index]
            branched_count_rel = 0.0
            branched_count_sum = sum(stats.branched_count[var.index])
            if branched_count_sum > 0
                branched_count_rel = branched_count / branched_count_sum
            end
            push!(
                features,
                Float32[
                    nfrac,
                    node.fractional_values[σ[i]],
                    node.depth,
                    pseudocost_scores[σ[i]][1],
                    branched_count,
                    branched_count_rel,
                ],
            )
        end
    end

    _set_node_bounds(node)
    no_improv_count, n_sb_calls = 0, 0
    max_score, max_var = (-Inf, -Inf), sorted_vars[1]
    for (i, var) in enumerate(sorted_vars)

        # Decide whether to use strong branching
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
            # Compute strong branching score
            n_sb_calls += 1
            score = _strong_branch_score(
                node = node,
                pool = pool,
                var = var,
                x = node.fractional_values[σ[i]],
                side_effect = rule.side_effect,
                max_iterations = rule.max_iterations,
                aggregation = rule.aggregation,
            )

            if rule.collect
                # Store training data
                push!(stats.score_var_names, name(node.mip, var))
                push!(stats.score_features, features[i])
                push!(stats.score_targets, score[1])
            end
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

    # Update statistics
    stats.branched_count[max_var.index] += 1
    stats.num_strong_branch_calls += n_sb_calls

    return max_var
end

function collect!(rule::ReliabilityBranching, h5)
    if rule.stats.num_strong_branch_calls == 0
        return
    end
    h5.put_array("bb_score_var_names", to_str_array(rule.stats.score_var_names))
    h5.put_array("bb_score_features", vcat(rule.stats.score_features'...))
    h5.put_array("bb_score_targets", rule.stats.score_targets)
end
