#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using SparseArrays

Base.@kwdef mutable struct ProblemData
    obj::Vector{Float64}
    obj_offset::Float64
    obj_sense::Any
    constr_lb::Vector{Float64}
    constr_ub::Vector{Float64}
    constr_lhs::SparseMatrixCSC
    var_lb::Vector{Float64}
    var_ub::Vector{Float64}
    var_types::Vector{Char}
    var_names::Vector{String}
end

Base.@kwdef mutable struct Tableau
    obj::Any
    lhs::Any
    rhs::Any
    z::Any
end

Base.@kwdef mutable struct Basis
    var_basic::Any
    var_nonbasic::Any
    constr_basic::Any
    constr_nonbasic::Any
end

Base.@kwdef mutable struct ConstraintSet
    lhs::SparseMatrixCSC
    ub::Vector{Float64}
    lb::Vector{Float64}
end

export ProblemData, Tableau, Basis, ConstraintSet
