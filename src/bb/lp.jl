#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import Base: values, convert
using Base.Threads
import Base.Threads: @threads, nthreads, threadid
using JuMP
using MathOptInterface
const MOI = MathOptInterface

function init(constructor)::MIP
    return MIP(constructor, Any[nothing for t = 1:nthreads()], Variable[], 1.0, 0)
end

function read!(mip::MIP, filename::AbstractString)::Nothing
    @threads for t = 1:nthreads()
        model = read_from_file(filename)
        set_optimizer(model, mip.constructor)
        mip.optimizers[t] = backend(model)
        _replace_zero_one!(mip.optimizers[t])
        if t == 1
            _assert_supported(mip.optimizers[t])
            mip.binary_variables = _get_binary_variables(mip.optimizers[t])
            mip.sense = _get_objective_sense(mip.optimizers[t])
        end
        _relax_integrality!(mip.optimizers[t])
        set_silent(model)
    end
    return
end

function _assert_supported(optimizer::MOI.AbstractOptimizer)::Nothing
    types = MOI.get(optimizer, MOI.ListOfConstraintTypesPresent())
    for (F, S) in types
        _assert_supported(F, S)
    end
end

function _assert_supported(F::Type, S::Type)::Nothing
    if F in [MOI.ScalarAffineFunction{Float64}, MOI.VariableIndex] && S in [
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
        MOI.EqualTo{Float64},
        MOI.Interval{Float64},
    ]
        return
    end
    if F in [MOI.VariableIndex] && S in [MOI.Integer, MOI.ZeroOne]
        return
    end
    error("MOI constraint not supported: $F in $S")
end

function _get_objective_sense(optimizer::MOI.AbstractOptimizer)::Float64
    sense = MOI.get(optimizer, MOI.ObjectiveSense())
    if sense == MOI.MIN_SENSE
        return 1.0
    elseif sense == MOI.MAX_SENSE
        return -1.0
    else
        error("objective sense not supported: $sense")
    end
end

_bounds_constraint(v::Variable) =
    MOI.ConstraintIndex{MOI.VariableIndex,MOI.Interval{Float64}}(v.index)

function _replace_zero_one!(optimizer::MOI.AbstractOptimizer)::Nothing
    constrs_to_delete = MOI.ConstraintIndex[]
    funcs = MOI.VariableIndex[]
    sets = Union{MOI.Interval,MOI.Integer}[]
    for ci in
        MOI.get(optimizer, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.ZeroOne}())
        func = MOI.get(optimizer, MOI.ConstraintFunction(), ci)
        var = func.value
        push!(constrs_to_delete, ci)
        push!(funcs, MOI.VariableIndex(var))
        push!(funcs, MOI.VariableIndex(var))
        push!(sets, MOI.Interval{Float64}(0.0, 1.0))
        push!(sets, MOI.Integer())
    end
    MOI.delete(optimizer, constrs_to_delete)
    MOI.add_constraints(optimizer, funcs, sets)
    return
end

function _get_binary_variables(optimizer::MOI.AbstractOptimizer)::Vector{Variable}
    vars = Variable[]
    for ci in
        MOI.get(optimizer, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}())
        func = MOI.get(optimizer, MOI.ConstraintFunction(), ci)
        var = Variable(func.value)

        MOI.is_valid(optimizer, _bounds_constraint(var)) ||
            error("$var is not interval-constrained")
        interval = MOI.get(optimizer, MOI.ConstraintSet(), _bounds_constraint(var))
        interval.lower == 0.0 || error("$var has lb != 0")
        interval.upper == 1.0 || error("$var has ub != 1")

        push!(vars, var)
    end
    return vars
end

function _relax_integrality!(optimizer::MOI.AbstractOptimizer)::Nothing
    indices =
        MOI.get(optimizer, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}())
    MOI.delete(optimizer, indices)
end

"""
    solve_relaxation(mip::MIP)::Tuple{Symbol, Float64}

Solve the linear relaxation of `mip` and returns a tuple containing the
solution status (either :Optimal or :Infeasible) and the optimal objective
value. If the problem is infeasible, the optimal value is Inf for minimization
problems and -Inf for maximization problems..
"""
function solve_relaxation!(mip::MIP)::Tuple{Symbol,Float64}
    t = threadid()
    MOI.optimize!(mip.optimizers[t])
    try
        mip.lp_iterations += MOI.get(mip.optimizers[t], MOI.SimplexIterations())
    catch
        # ignore
    end
    status = MOI.get(mip.optimizers[t], MOI.TerminationStatus())
    if status == MOI.OPTIMAL
        obj = MOI.get(mip.optimizers[t], MOI.ObjectiveValue())
        return :Optimal, obj * mip.sense
    elseif status in [MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED]
        return :Infeasible, Inf * mip.sense
    end
    error("unknown status: $status")
end

"""
    values(mip::MIP, vars::Vector{Variable})::Array{Float64}

Returns a vector `vals` which describes the current primal values for the
decision variables `vars`. More specifically, `vals[j]` is the current
primal value of `vars[j]`.
"""
function values(mip::MIP, vars::Vector{Variable})::Array{Float64}
    return MOI.get(
        mip.optimizers[threadid()],
        MOI.VariablePrimal(),
        [MOI.VariableIndex(v.index) for v in vars],
    )
end

values(mip::MIP) =
    values(mip, MOI.get(mip.optimizers[threadid()], MOI.ListOfVariableIndices()))

"""
    set_bounds!(mip::MIP,
                vars::Vector{Variable},
                lb::Array{Float64},
                ub::Array{Float64})

Modify the bounds of the given variables. More specifically, sets
upper and lower bounds of `vars[j]` to `ub[j]` and `lb[j]`, respectively.
"""
function set_bounds!(
    mip::MIP,
    vars::Vector{Variable},
    lb::Array{Float64},
    ub::Array{Float64},
)::Nothing
    t = threadid()
    MOI.delete(mip.optimizers[t], _bounds_constraint.(vars))
    funcs = MOI.VariableIndex[]
    sets = MOI.Interval[]
    for j = 1:length(vars)
        push!(funcs, MOI.VariableIndex(vars[j].index))
        push!(sets, MOI.Interval(lb[j], ub[j]))
    end
    MOI.add_constraints(mip.optimizers[t], funcs, sets)
    return
end

"""
    name(mip::MIP, var::Variable)::String

Return the name of the decision variable `var`.
"""
function name(mip::MIP, var::Variable)::String
    t = threadid()
    return MOI.get(mip.optimizers[t], MOI.VariableName(), MOI.VariableIndex(var.index))
end

convert(::Type{MOI.VariableIndex}, v::Variable) = MOI.VariableIndex(v.index)

"""
    probe(mip::MIP, var)::Tuple{Float64, Float64}

Suppose that the LP relaxation of `mip` has been solved and that `var` holds
a fractional value `f`. This function returns two numbers corresponding,
respectively, to the the optimal values of the LP relaxations having the
constraints var=1 and var=0 enforced. If any branch is infeasible, the optimal
value for that branch is Inf for minimization problems and -Inf for maximization
problems.
"""
function probe(mip::MIP, var)::Tuple{Float64,Float64}
    set_bounds!(mip, [var], [1.0], [1.0])
    status_up, obj_up = solve_relaxation!(mip)
    set_bounds!(mip, [var], [0.0], [0.0])
    status_down, obj_down = solve_relaxation!(mip)
    set_bounds!(mip, [var], [0.0], [1.0])
    return obj_up * mip.sense, obj_down * mip.sense
end
