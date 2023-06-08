#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using LinearAlgebra
using TimerOutputs

abstract type Transform end

function _isbounded(x)
    isfinite(x) || return false
    abs(x) < 1e15 || return false
    return true
end

function backwards!(transforms::Vector{Transform}, m::ConstraintSet; tol = 1e-8)
    for t in reverse(transforms)
        backwards!(t, m)
    end
    for (idx, val) in enumerate(m.lhs.nzval)
        if abs(val) < tol
            m.lhs.nzval[idx] = 0
        end
    end
end

function backwards(transforms::Vector{Transform}, m::ConstraintSet; tol = 1e-8)
    m2 = deepcopy(m)
    backwards!(transforms, m2; tol)
    return m2
end

function forward(transforms::Vector{Transform}, p::Vector{Float64})::Vector{Float64}
    for t in transforms
        p = forward(t, p)
    end
    return p
end

# -----------------------------------------------------------------------------

Base.@kwdef mutable struct ShiftVarLowerBoundsToZero <: Transform
    lb::Vector{Float64} = []
end

function forward!(t::ShiftVarLowerBoundsToZero, data::ProblemData)
    t.lb = copy(data.var_lb)
    data.obj_offset += dot(data.obj, t.lb)
    data.var_ub -= t.lb
    data.var_lb -= t.lb
    data.constr_lb -= data.constr_lhs * t.lb
    data.constr_ub -= data.constr_lhs * t.lb
end

function backwards!(t::ShiftVarLowerBoundsToZero, c::ConstraintSet)
    c.lb += c.lhs * t.lb
    c.ub += c.lhs * t.lb
end

function forward(t::ShiftVarLowerBoundsToZero, p::Vector{Float64})::Vector{Float64}
    return p - t.lb
end

# -----------------------------------------------------------------------------

Base.@kwdef mutable struct MoveVarUpperBoundsToConstrs <: Transform end

function forward!(t::MoveVarUpperBoundsToConstrs, data::ProblemData)
    _, ncols = size(data.constr_lhs)
    data.constr_lhs = [data.constr_lhs; I]
    data.constr_lb = [data.constr_lb; [-Inf for _ = 1:ncols]]
    data.constr_ub = [data.constr_ub; data.var_ub]
    data.var_ub .= Inf
end

function backwards!(::MoveVarUpperBoundsToConstrs, ::ConstraintSet)
    # nop
end

function forward(t::MoveVarUpperBoundsToConstrs, p::Vector{Float64})::Vector{Float64}
    return p
end

# -----------------------------------------------------------------------------

Base.@kwdef mutable struct AddSlackVariables <: Transform
    M1::SparseMatrixCSC = spzeros(0)
    M2::Vector{Float64} = []
    ncols_orig::Int = 0
    GE::Int = 0
    LE::Int = 0
    lhs_ge::SparseMatrixCSC = spzeros(0)
    lhs_le::SparseMatrixCSC = spzeros(0)
    rhs_le::Vector{Float64} = []
    rhs_ge::Vector{Float64} = []
end

function forward!(t::AddSlackVariables, data::ProblemData)
    nrows, ncols = size(data.constr_lhs)
    isequality = abs.(data.constr_ub .- data.constr_lb) .< 1e-6
    eq = [i for i = 1:nrows if isequality[i]]
    ge = [i for i = 1:nrows if isfinite(data.constr_lb[i]) && !isequality[i]]
    le = [i for i = 1:nrows if isfinite(data.constr_ub[i]) && !isequality[i]]
    EQ, GE, LE = length(eq), length(ge), length(le)

    t.M1 = [
        I spzeros(ncols, GE + LE)
        data.constr_lhs[ge, :] spzeros(GE, GE + LE)
        -data.constr_lhs[le, :] spzeros(LE, GE + LE)
    ]
    t.M2 = [
        zeros(ncols)
        data.constr_lb[ge]
        -data.constr_ub[le]
    ]
    t.ncols_orig = ncols
    t.GE, t.LE = GE, LE
    t.lhs_ge = data.constr_lhs[ge, :]
    t.lhs_le = data.constr_lhs[le, :]
    t.rhs_ge = data.constr_lb[ge]
    t.rhs_le = data.constr_ub[le]

    data.constr_lhs = [
        data.constr_lhs[eq, :] spzeros(EQ, GE) spzeros(EQ, LE)
        data.constr_lhs[ge, :] -I spzeros(GE, LE)
        data.constr_lhs[le, :] spzeros(LE, GE) I
    ]
    data.obj = [data.obj; zeros(GE + LE)]
    data.var_lb = [data.var_lb; zeros(GE + LE)]
    data.var_ub = [data.var_ub; [Inf for _ = 1:(GE+LE)]]
    data.var_names = [data.var_names; ["__s$i" for i = 1:(GE+LE)]]
    data.var_types = [data.var_types; ['C' for _ = 1:(GE+LE)]]
    data.constr_lb = [
        data.constr_lb[eq]
        data.constr_lb[ge]
        data.constr_ub[le]
    ]
    data.constr_ub = copy(data.constr_lb)
end

function backwards!(t::AddSlackVariables, c::ConstraintSet)
    c.lb += c.lhs * t.M2
    c.ub += c.lhs * t.M2
    c.lhs = (c.lhs*t.M1)[:, 1:t.ncols_orig]
end

function forward(t::AddSlackVariables, x::Vector{Float64})::Vector{Float64}
    return [
        x
        t.lhs_ge * x - t.rhs_ge
        t.rhs_le - t.lhs_le * x
    ]
end

# -----------------------------------------------------------------------------

Base.@kwdef mutable struct SplitFreeVars <: Transform
    F::Int = 0
    B::Int = 0
    free::Vector{Int} = []
    others::Vector{Int} = []
end

function forward!(t::SplitFreeVars, data::ProblemData)
    lhs = data.constr_lhs
    _, ncols = size(lhs)
    free = [i for i = 1:ncols if !isfinite(data.var_lb[i]) && !isfinite(data.var_ub[i])]
    others = [i for i = 1:ncols if isfinite(data.var_lb[i]) || isfinite(data.var_ub[i])]
    t.F = length(free)
    t.B = length(others)
    t.free, t.others = free, others
    data.obj = [
        data.obj[others]
        data.obj[free]
        -data.obj[free]
    ]
    data.constr_lhs = [lhs[:, others] lhs[:, free] -lhs[:, free]]
    data.var_lb = [
        data.var_lb[others]
        [0.0 for _ in free]
        [0.0 for _ in free]
    ]
    data.var_ub = [
        data.var_ub[others]
        [Inf for _ in free]
        [Inf for _ in free]
    ]
    data.var_types = [
        data.var_types[others]
        data.var_types[free]
        data.var_types[free]
    ]
    data.var_names = [
        data.var_names[others]
        ["$(v)_p" for v in data.var_names[free]]
        ["$(v)_m" for v in data.var_names[free]]
    ]
end

function backwards!(t::SplitFreeVars, c::ConstraintSet)
    # Convert GE constraints into LE
    nrows, _ = size(c.lhs)
    ge = [i for i = 1:nrows if isfinite(c.lb[i])]
    c.ub[ge], c.lb[ge] = -c.lb[ge], -c.ub[ge]
    c.lhs[ge, :] *= -1

    # Assert only LE constraints are left (EQ constraints are not supported)
    @assert all(c.lb .== -Inf)

    # Take minimum (weakest) coefficient
    B, F = t.B, t.F
    for i = 1:F
        c.lhs[:, B+i] = min.(c.lhs[:, B+i], -c.lhs[:, B+F+i])
    end
    c.lhs = c.lhs[:, 1:(B+F)]
end

function forward(t::SplitFreeVars, p::Vector{Float64})::Vector{Float64}
    return [
        p[t.others]
        max.(p[t.free], 0)
        max.(-p[t.free], 0)
    ]
end

# -----------------------------------------------------------------------------

Base.@kwdef mutable struct FlipUnboundedBelowVars <: Transform
    flip_idx::Vector{Int} = []
end

function forward!(t::FlipUnboundedBelowVars, data::ProblemData)
    _, ncols = size(data.constr_lhs)
    for i = 1:ncols
        if isfinite(data.var_lb[i])
            continue
        end
        data.obj[i] *= -1
        data.constr_lhs[:, i] *= -1
        data.var_lb[i], data.var_ub[i] = -data.var_ub[i], -data.var_lb[i]
        push!(t.flip_idx, i)
    end
end

function backwards!(t::FlipUnboundedBelowVars, c::ConstraintSet)
    for i in t.flip_idx
        c.lhs[:, i] *= -1
    end
end

function forward(t::FlipUnboundedBelowVars, p::Vector{Float64})::Vector{Float64}
    p2 = copy(p)
    p2[t.flip_idx] *= -1
    return p2
end

# -----------------------------------------------------------------------------

function _assert_standard_form(data::ProblemData)
    # Check sizes
    nrows, ncols = size(data.constr_lhs)
    @assert length(data.constr_lb) == nrows
    @assert length(data.constr_ub) == nrows
    @assert length(data.obj) == ncols
    @assert length(data.var_lb) == ncols
    @assert length(data.var_ub) == ncols
    @assert length(data.var_names) == ncols
    @assert length(data.var_types) == ncols

    # Check standard form
    @assert all(data.var_lb .== 0.0)
    @assert all(data.var_ub .== Inf)
    @assert all(data.constr_lb .== data.constr_ub)
end

function convert_to_standard_form!(data::ProblemData)::Vector{Transform}
    transforms = []
    function _apply!(t)
        push!(transforms, t)
        forward!(t, data)
    end
    @timeit "Split free vars" begin
        _apply!(SplitFreeVars())
    end
    @timeit "Flip unbounded-below vars" begin
        _apply!(FlipUnboundedBelowVars())
    end
    @timeit "Shift var lower bounds to zero" begin
        _apply!(ShiftVarLowerBoundsToZero())
    end
    @timeit "Move var upper bounds to constrs" begin
        _apply!(MoveVarUpperBoundsToConstrs())
    end
    @timeit "Add slack vars" begin
        _apply!(AddSlackVariables())
    end
    _assert_standard_form(data)
    return transforms
end

function convert_to_standard_form(data::ProblemData)::Tuple{ProblemData,Vector{Transform}}
    data2 = deepcopy(data)
    transforms = convert_to_standard_form!(data2)
    return (data2, transforms)
end

export convert_to_standard_form!,
    convert_to_standard_form,
    forward!,
    backwards!,
    backwards,
    AddSlackVariables,
    SplitFreeVars,
    forward
