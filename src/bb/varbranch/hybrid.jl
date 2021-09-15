#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

"""
    HybridBranching(depth_cutoff::Int,
                    shallow_rule::VariableBranchingRule,
                    deep_rule::::VariableBranchingRule)

Branching strategy that switches between two variable branching strategies,
according to the depth of the node.

More specifically, if `node.depth <= depth_cutoff`, then `shallow_rule` is
applied. Otherwise, `deep_rule` is applied.
"""
mutable struct HybridBranching <: VariableBranchingRule
    depth_cutoff::Int
    shallow_rule::VariableBranchingRule
    deep_rule::VariableBranchingRule
end

HybridBranching(depth_cutoff::Int = 10) =
    HybridBranching(depth_cutoff, StrongBranching(), PseudocostBranching())

function find_branching_var(rule::HybridBranching, node::Node, pool::NodePool)::Variable
    if node.depth <= rule.depth_cutoff
        return find_branching_var(rule.shallow_rule, node, pool)
    else
        return find_branching_var(rule.deep_rule, node, pool)
    end
end
