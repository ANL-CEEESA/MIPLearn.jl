#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import Base: values, convert
using Base.Threads
import Base.Threads: @threads, nthreads, threadid
using JuMP
using MathOptInterface
const MOI = MathOptInterface

function init(constructor)::MIP
    return MIP(
        constructor = constructor,
        optimizers = Any[nothing for t = 1:nthreads()],
        int_vars = Variable[],
        int_vars_lb = Float64[],
        int_vars_ub = Float64[],
        sense = 1.0,
        lp_iterations = 0,
        nvars = 0,
    )
end

function read!(mip::MIP, filename::AbstractString)::Nothing
    load!(mip, read_from_file(filename))
    return
end

function load!(mip::MIP, prototype::JuMP.Model)
    mip.nvars = num_variables(prototype)
    _replace_zero_one!(backend(prototype))
    _assert_supported(backend(prototype))
    mip.int_vars, mip.int_vars_lb, mip.int_vars_ub = _get_int_variables(backend(prototype))
    mip.sense = _get_objective_sense(backend(prototype))
    _relax_integrality!(backend(prototype))
    @threads for t = 1:nthreads()
        model = Model(mip.constructor)
        MOI.copy_to(model, backend(prototype))
        mip.optimizers[t] = backend(model)
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

_interval_index(v::Variable) =
    MOI.ConstraintIndex{MOI.VariableIndex,MOI.Interval{Float64}}(v.index)

_lower_bound_index(v::Variable) =
    MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}}(v.index)

_upper_bound_index(v::Variable) =
    MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}}(v.index)


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

function _get_int_variables(
    optimizer::MOI.AbstractOptimizer,
)::Tuple{Vector{Variable},Vector{Float64},Vector{Float64}}
    vars = Variable[]
    lb = Float64[]
    ub = Float64[]
    for ci in
        MOI.get(optimizer, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Integer}())
        func = MOI.get(optimizer, MOI.ConstraintFunction(), ci)
        var = Variable(func.value)

        var_lb, var_ub = -Inf, Inf
        if MOI.is_valid(optimizer, _interval_index(var))
            constr = MOI.get(optimizer, MOI.ConstraintSet(), _interval_index(var))
            var_ub = constr.upper
            var_lb = constr.lower
        else
            # If interval constraint is not found, query individual lower/upper bound
            # constraints and replace them by an interval constraint.
            if MOI.is_valid(optimizer, _lower_bound_index(var))
                constr = MOI.get(optimizer, MOI.ConstraintSet(), _lower_bound_index(var))
                var_lb = constr.lower
                MOI.delete(optimizer, _lower_bound_index(var))
            end
            if MOI.is_valid(optimizer, _upper_bound_index(var))
                constr = MOI.get(optimizer, MOI.ConstraintSet(), _upper_bound_index(var))
                var_ub = constr.upper
                MOI.delete(optimizer, _upper_bound_index(var))
            end
            MOI.add_constraint(optimizer, var, MOI.Interval(var_lb, var_ub))
        end
        push!(vars, var)
        push!(lb, var_lb)
        push!(ub, var_ub)
    end
    return vars, lb, ub
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
    for j = 1:length(vars)
        MOI.delete(mip.optimizers[t], _interval_index(vars[j]))
        MOI.add_constraint(
            mip.optimizers[t],
            MOI.VariableIndex(vars[j].index),
            MOI.Interval(lb[j], ub[j]),
        )
    end
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

"""
    probe(mip::MIP, var, x, lb, ub, max_iterations)::Tuple{Float64, Float64}

Suppose that the LP relaxation of `mip` has been solved and that `var` holds
a fractional value `x`. This function returns two numbers corresponding,
respectively, to the the optimal values of the LP relaxations having the
constraints `ceil(x) <= var <= ub` and `lb <= var <= floor(x)` enforced.
If any branch is infeasible, the optimal value for that branch is Inf for
minimization problems and -Inf for maximization problems.
"""
function probe(
    mip::MIP,
    var::Variable,
    x::Float64,
    lb::Float64,
    ub::Float64,
    max_iterations::Int,
)::Tuple{Float64,Float64}
    return _probe(mip, mip.optimizers[threadid()], var, x, lb, ub, max_iterations)
end

function _probe(
    mip::MIP,
    _,
    var::Variable,
    x::Float64,
    lb::Float64,
    ub::Float64,
    ::Int,
)::Tuple{Float64,Float64}
    set_bounds!(mip, [var], [ceil(x)], [ceil(x)])
    _, obj_up = solve_relaxation!(mip)
    set_bounds!(mip, [var], [floor(x)], [floor(x)])
    _, obj_down = solve_relaxation!(mip)
    set_bounds!(mip, [var], [lb], [ub])
    return obj_up * mip.sense, obj_down * mip.sense
end
