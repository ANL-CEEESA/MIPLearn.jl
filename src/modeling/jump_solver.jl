#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using MathOptInterface
const MOI = MathOptInterface
using TimerOutputs


mutable struct JuMPSolverData
    optimizer_factory
    varname_to_var::Dict{AbstractString,VariableRef}
    cname_to_constr::Dict{AbstractString,JuMP.ConstraintRef}
    instance::Union{Nothing,PyObject}
    model::Union{Nothing,JuMP.Model}
    bin_vars::Vector{JuMP.VariableRef}
    solution::Vector{Float64}
    reduced_costs::Vector{Float64}
    dual_values::Dict{JuMP.ConstraintRef,Float64}
end


"""
    _optimize_and_capture_output!(model; tee=tee)

Optimizes a given JuMP model while capturing the solver log, then returns that log.
If tee=true, prints the solver log to the standard output as the optimization takes place.
"""
function _optimize_and_capture_output!(model; tee::Bool=false)
    logname = tempname()
    logfile = open(logname, "w")
    redirect_stdout(logfile) do 
        JuMP.optimize!(model)
        Base.Libc.flush_cstdio()
    end
    close(logfile)
    log = String(read(logname))
    rm(logname)
    if tee
        println(log)
    end
    return log
end


function _update_solution!(data::JuMPSolverData)
    vars = JuMP.all_variables(data.model)
    data.solution = [JuMP.value(var) for var in vars]

    # Reduced costs
    if has_duals(data.model)
        data.reduced_costs = []
        for var in vars
            rc = 0.0
            if has_upper_bound(var)
                rc += shadow_price(UpperBoundRef(var))
            end
            if has_lower_bound(var)
                # FIXME: Remove negative sign
                rc -= shadow_price(LowerBoundRef(var))
            end
            if is_fixed(var)
                rc += shadow_price(FixRef(var))
            end
            push!(data.reduced_costs, rc)
        end

        data.dual_values = Dict()
        for (ftype, stype) in JuMP.list_of_constraint_types(data.model)
            for constr in JuMP.all_constraints(data.model, ftype, stype)
                # FIXME: Remove negative sign
                data.dual_values[constr] = -JuMP.dual(constr)
            end
        end
    else
        data.reduced_costs = []
        data.dual_values = Dict()
    end
end


function solve(
    data::JuMPSolverData;
    tee::Bool=false,
    iteration_cb=nothing,
)
    model = data.model
    wallclock_time = 0
    log = ""
    while true
        log *= _optimize_and_capture_output!(model, tee=tee)
        wallclock_time += JuMP.solve_time(model)
        if iteration_cb !== nothing
            iteration_cb() || break
        else
            break
        end
    end
    _update_solution!(data)
    primal_bound = JuMP.objective_value(model)
    dual_bound = JuMP.objective_bound(model)
    if JuMP.objective_sense(model) == MOI.MIN_SENSE
        sense = "min"
        lower_bound = dual_bound
        upper_bound = primal_bound
    else
        sense = "max"
        lower_bound = primal_bound
        upper_bound = dual_bound
    end
    return miplearn.solvers.internal.MIPSolveStats(
        mip_lower_bound=lower_bound,
        mip_upper_bound=upper_bound,
        mip_sense=sense,
        mip_wallclock_time=wallclock_time,
        mip_nodes=1,
        mip_log=log,
        mip_warm_start_value=nothing,
    )
end


function solve_lp(data::JuMPSolverData; tee::Bool=false)
    model, bin_vars = data.model, data.bin_vars
    for var in bin_vars
        JuMP.unset_binary(var)
        JuMP.set_upper_bound(var, 1.0)
        JuMP.set_lower_bound(var, 0.0)
    end
    wallclock_time = @elapsed begin
        log = _optimize_and_capture_output!(model, tee=tee)
    end
    _update_solution!(data)
    obj_value = JuMP.objective_value(model)
    for var in bin_vars
        JuMP.set_binary(var)
    end
    return miplearn.solvers.internal.LPSolveStats(
        lp_value=obj_value,
        lp_log=log,
        lp_wallclock_time=wallclock_time,
    )
end


function set_instance!(data::JuMPSolverData, instance, model::JuMP.Model)
    data.instance = instance
    data.model = model
    data.bin_vars = [
        var
        for var in JuMP.all_variables(data.model)
        if JuMP.is_binary(var)
    ]
    data.varname_to_var = Dict(
        JuMP.name(var) => var
        for var in JuMP.all_variables(data.model)
    )
    JuMP.set_optimizer(model, data.optimizer_factory)
    data.cname_to_constr = Dict()
    for (ftype, stype) in JuMP.list_of_constraint_types(model)
        for constr in JuMP.all_constraints(model, ftype, stype)
            name = JuMP.name(constr)
            length(name) > 0 || continue
            data.cname_to_constr[name] = constr
        end
    end
end


function fix!(data::JuMPSolverData, solution)
    for (varname, value) in solution
        value !== nothing || continue
        var = data.varname_to_var[varname]
        JuMP.fix(var, value, force=true)
    end
end


function set_warm_start!(data::JuMPSolverData, solution)
    for (varname, value) in solution
        value !== nothing || continue
        var = data.varname_to_var[varname]
        JuMP.set_start_value(var, value)
    end
end


function is_infeasible(data::JuMPSolverData)
    return JuMP.termination_status(data.model) == MOI.INFEASIBLE
end


function get_variables(
    data::JuMPSolverData;
    with_static::Bool,
)
    vars = JuMP.all_variables(data.model)
    lb, ub, types, obj_coeffs = nothing, nothing, nothing, nothing
    rc = nothing

    # Variable names
    names = Tuple(JuMP.name.(vars))

    if with_static
        # Lower bounds
        lb = Tuple(
            JuMP.is_binary(v) ? 0.0 :
                JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) :
                    -Inf
            for v in vars
        )

        # Upper bounds
        ub = Tuple(
            JuMP.is_binary(v) ? 1.0 :
                JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) :
                    Inf
            for v in vars
        )

        # Variable types
        types = Tuple(
            JuMP.is_binary(v) ? "B" :
                JuMP.is_integer(v) ? "I" :
                    "C"
            for v in vars
        )

        # Objective function coefficients
        obj = objective_function(data.model)
        obj_coeffs = Tuple(
            v ∈ keys(obj.terms) ? obj.terms[v] : 0.0
            for v in vars
        )
    end

    rc = isempty(data.reduced_costs) ? nothing : Tuple(data.reduced_costs)
    values = isempty(data.solution) ? nothing : Tuple(data.solution)

    return miplearn.features.VariableFeatures(
        names=names,
        lower_bounds=lb,
        upper_bounds=ub,
        types=types,
        obj_coeffs=obj_coeffs,
        reduced_costs=rc,
        values=values,
    )
end


function get_constraints(
    data::JuMPSolverData;
    with_static::Bool,
)
    names = []
    senses, lhs, rhs = nothing, nothing, nothing
    dual_values = nothing

    if !isempty(data.dual_values)
        dual_values = []
    end

    if with_static
        senses, lhs, rhs = [], [], []
    end

    for (ftype, stype) in JuMP.list_of_constraint_types(data.model)
        ftype in [JuMP.AffExpr, JuMP.VariableRef] || error("Unsupported constraint type: ($ftype, $stype)")
        for constr in JuMP.all_constraints(data.model, ftype, stype)
            cset = MOI.get(
                constr.model.moi_backend,
                MOI.ConstraintSet(),
                constr.index,
            )
            name = JuMP.name(constr)
            length(name) > 0 || continue
            push!(names, name)

            if !isempty(data.dual_values)
                push!(dual_values, data.dual_values[constr])
            end

            if with_static
                if ftype == JuMP.AffExpr
                    push!(
                        lhs,
                        Tuple(
                            (
                                MOI.get(
                                    constr.model.moi_backend,
                                    MOI.VariableName(),
                                    term.variable_index
                                ),
                                term.coefficient,
                            )
                            for term in MOI.get(
                                constr.model.moi_backend,
                                MOI.ConstraintFunction(),
                                constr.index,
                            ).terms
                        )
                    )
                    if stype == MOI.EqualTo{Float64}
                        push!(senses, "=")
                        push!(rhs, cset.value)
                    elseif stype == MOI.LessThan{Float64}
                        push!(senses, "<")
                        push!(rhs, cset.upper)
                    elseif stype == MOI.GreaterThan{Float64}
                        push!(senses, ">")
                        push!(rhs, cset.lower)
                    else
                        error("Unsupported set: $stype")
                    end
                else
                    error("Unsupported ftype: $ftype")
                end
            end
        end
    end

    function to_tuple(x)
        x !== nothing || return nothing
        return Tuple(x)
    end

    return miplearn.features.ConstraintFeatures(
        names=to_tuple(names),
        senses=to_tuple(senses),
        lhs=to_tuple(lhs),
        rhs=to_tuple(rhs),
        dual_values=to_tuple(dual_values),
    )
end


function build_test_instance_knapsack()
    weights = [23.0, 26.0, 20.0, 18.0]
    prices = [505.0, 352.0, 458.0, 220.0]
    capacity = 67.0

    model = Model()
    n = length(weights)
    @variable(model, x[0:n-1], Bin)
    @variable(model, z, lower_bound=0.0, upper_bound=capacity)
    @objective(model, Max, sum(x[i-1] * prices[i] for i in 1:n))
    @constraint(model, eq_capacity, sum(x[i-1] * weights[i] for i in 1:n) - z == 0)

    return JuMPInstance(model)
end


@pydef mutable struct JuMPSolver <: miplearn.solvers.internal.InternalSolver
    function __init__(self, optimizer_factory)
        self.data = JuMPSolverData(
            optimizer_factory,
            Dict(),  # varname_to_var
            Dict(),  # cname_to_constr
            nothing,  # instance
            nothing,  # model
            [],  # bin_vars
            [],  # solution
            [],  # reduced_costs
            Dict(),  # dual_values
        )
    end

    add_constraints(self, cf) =
        error("not implemented")

    are_constraints_satisfied(self, cf; tol=1e-5) =
        error("not implemented")

    build_test_instance_infeasible(self) =
        error("not implemented")

    build_test_instance_knapsack(self) =
        build_test_instance_knapsack()
    
    # FIXME: Actually clone instead of returning self
    clone(self) = self

    fix(self, solution) =
        fix!(self.data, solution)
    
    get_solution(self) =
        self.data.solution

    get_constraints(
        self;
        with_static=true,
        with_sa=true,
        with_lhs=true,
    ) = get_constraints(
        self.data,
        with_static=with_static,
    )

    get_constraint_attrs(self) = [
        # "basis_status",
        "categories",
        "dual_values",
        "lazy",
        "lhs",
        "names",
        "rhs",
        # "sa_rhs_down",
        # "sa_rhs_up",
        "senses",
        # "slacks",
        "user_features",
    ]
    
    get_variables(self; with_static=true) =
        get_variables(self.data; with_static=with_static)
    
    get_variable_attrs(self) =  [
        "names",
        # "basis_status",
        "categories",
        "lower_bounds",
        "obj_coeffs",
        "reduced_costs",
        # "sa_lb_down",
        # "sa_lb_up",
        # "sa_obj_down",
        # "sa_obj_up",
        # "sa_ub_down",
        # "sa_ub_up",
        "types",
        "upper_bounds",
        "user_features",
        "values",
    ]

    is_infeasible(self) =
        is_infeasible(self.data)

    remove_constraints(self, names) =
        error("not implemented")

    set_instance(self, instance, model) =
        set_instance!(self.data, instance, model)
    
    set_warm_start(self, solution) =
        set_warm_start!(self.data, solution)

    solve(
        self;
        tee=false,
        iteration_cb=nothing,
        lazy_cb=nothing,
        user_cut_cb=nothing,
    ) = solve(
        self.data,
        tee=tee,
        iteration_cb=iteration_cb,
    )
    
    solve_lp(self; tee=false) =
        solve_lp(self.data, tee=tee)
end


export JuMPSolver
