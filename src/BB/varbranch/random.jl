#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Random


"""
    RandomBranching()

Branching strategy that picks a fractional variable randomly.
"""
struct RandomBranching <: VariableBranchingRule end

function find_branching_var(rule::RandomBranching, node::Node, pool::NodePool)::Variable
    return shuffle(node.fractional_variables)[1]
end
