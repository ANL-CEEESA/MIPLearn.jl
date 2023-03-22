#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

"""
    FirstInfeasibleBranching()

Branching rule that always selects the first fractional variable.
"""
struct FirstInfeasibleBranching <: VariableBranchingRule end

function find_branching_var(
    rule::FirstInfeasibleBranching,
    node::Node,
    pool::NodePool,
)::Variable
    return node.fractional_variables[1]
end

"""
    LeastInfeasibleBranching()

Branching strategy that select the fractional variable whose value is the closest
to an integral value.
"""
struct LeastInfeasibleBranching <: VariableBranchingRule end

function find_branching_var(
    rule::LeastInfeasibleBranching,
    node::Node,
    pool::NodePool,
)::Variable
    scores = [max(v - floor(v), ceil(v) - v) for v in node.fractional_values]
    _, max_offset = findmax(scores)
    return node.fractional_variables[max_offset]
end

"""
    MostInfeasibleBranching()

Branching strategy that selects the fractional variable whose value is closest
to 1/2.
"""
struct MostInfeasibleBranching <: VariableBranchingRule end

function find_branching_var(
    rule::MostInfeasibleBranching,
    node::Node,
    pool::NodePool,
)::Variable
    scores = [min(v - floor(v), ceil(v) - v) for v in node.fractional_values]
    _, max_offset = findmax(scores)
    return node.fractional_variables[max_offset]
end
