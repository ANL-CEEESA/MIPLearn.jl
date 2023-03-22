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
    rb_rule = ReliabilityBranching(
        min_samples = typemax(Int),
        max_sb_calls = rule.max_calls,
        look_ahead = rule.look_ahead,
        side_effect = rule.side_effect,
        max_iterations = rule.max_iterations,
        aggregation = rule.aggregation,
    )
    return find_branching_var(rb_rule, node, pool)
end
