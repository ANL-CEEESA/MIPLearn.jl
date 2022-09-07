#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using DataStructures
abstract type VariableBranchingRule end

struct Variable
    index::Any
end

mutable struct MIP
    constructor::Any
    optimizers::Vector
    int_vars::Vector{Variable}
    int_vars_lb::Vector{Float64}
    int_vars_ub::Vector{Float64}
    sense::Float64
    lp_iterations::Int64
end

struct Node
    mip::MIP
    index::Int
    depth::Int
    obj::Float64
    status::Symbol
    branch_vars::Array{Variable}
    branch_lb::Array{Float64}
    branch_ub::Array{Float64}
    fractional_variables::Array{Variable}
    fractional_values::Array{Float64}
    parent::Union{Nothing,Node}
end

Base.@kwdef mutable struct History
    avg_pseudocost_up::Float64 = 1.0
    avg_pseudocost_down::Float64 = 1.0
end

mutable struct VariableHistory
    fractional_values::Array{Float64}
    obj_change_up::Array{Float64}
    obj_change_down::Array{Float64}
    obj_ratio_up::Array{Float64}
    obj_ratio_down::Array{Float64}
    pseudocost_up::Float64
    pseudocost_down::Float64

    VariableHistory() = new(
        Float64[], # fractional_values
        Float64[], # obj_change_up
        Float64[], # obj_change_down
        Float64[], # obj_ratio_up
        Float64[], # obj_ratio_up
        0.0, # pseudocost_up
        0.0, # pseudocost_down
    )
end

Base.@kwdef mutable struct NodePool
    mip::MIP
    pending::PriorityQueue{Node,Float64} = PriorityQueue{Node,Float64}()
    processing::PriorityQueue{Node,Float64} = PriorityQueue{Node,Float64}()
    processed::Int = 0
    next_index::Int = 1
    lock::ReentrantLock = ReentrantLock()
    primal_bound::Float64 = Inf
    dual_bound::Float64 = Inf
    gap::Float64 = Inf
    history::History = History()
    var_history::Dict{Variable,VariableHistory} = Dict()
end
