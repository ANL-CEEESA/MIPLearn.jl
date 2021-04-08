#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using MathOptInterface
const MOI = MathOptInterface
using TimerOutputs


mutable struct JuMPSolverData
    varname_to_var
    optimizer
    instance
    model
    bin_vars
    solution::Union{Nothing, Dict{String,Float64}}
    cname_to_constr
end


"""
    optimize_and_capture_output!(model; tee=tee)

Optimizes a given JuMP model while capturing the solver log, then returns that log.
If tee=true, prints the solver log to the standard output as the optimization takes place.
"""
function optimize_and_capture_output!(model; tee::Bool=false)
    original_stdout = stdout
    rd, wr = redirect_stdout()
    task = @async begin
        log = ""
        while true
            line = String(readavailable(rd))
            isopen(rd) || break
            log *= String(line)
            if tee
                print(original_stdout, line)
                flush(original_stdout)
            end
        end
        return log
    end
    JuMP.optimize!(model)
    sleep(1)
    redirect_stdout(original_stdout)
    close(rd)
    return fetch(task)
end


function solve(
    data::JuMPSolverData;
    tee::Bool=false,
    iteration_cb,
)::Dict
    instance, model = data.instance, data.model
    wallclock_time = 0
    log = ""
    while true
        log *= optimize_and_capture_output!(model, tee=tee)
        wallclock_time += JuMP.solve_time(model)
        if iteration_cb !== nothing
            iteration_cb() || break
        else
            break
        end
    end
    update_solution!(data)
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
    return Dict(
        "Lower bound" => lower_bound,
        "Upper bound" => upper_bound,
        "Sense" => sense,
        "Wallclock time" => wallclock_time,
        "Nodes" => 1,
        "MIP log" => log,
        "Warm start value" => nothing,
    )
end


function solve_lp(data::JuMPSolverData; tee::Bool=false)
    model, bin_vars = data.model, data.bin_vars
    for var in bin_vars
        JuMP.unset_binary(var)
        JuMP.set_upper_bound(var, 1.0)
        JuMP.set_lower_bound(var, 0.0)
    end
    log = optimize_and_capture_output!(model, tee=tee)
    update_solution!(data)
    obj_value = JuMP.objective_value(model)
    for var in bin_vars
        JuMP.set_binary(var)
    end
    return Dict(
        "LP value" => obj_value,
        "LP log" => log,
    )
end


function update_solution!(data::JuMPSolverData)
    data.solution = Dict(
        JuMP.name(var) => JuMP.value(var)
        for var in JuMP.all_variables(data.model)
    )
end


function set_instance!(data::JuMPSolverData, instance, model)
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
    if data.optimizer !== nothing
        JuMP.set_optimizer(model, data.optimizer)
    end
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


function get_variable_names(data::JuMPSolverData)
    return [JuMP.name(var) for var in JuMP.all_variables(data.model)]
end


function is_infeasible(data::JuMPSolverData)
    return JuMP.termination_status(data.model) == MOI.INFEASIBLE
end


function get_constraint_ids(data::JuMPSolverData)
    return [cname for cname in keys(data.cname_to_constr)]
end


function get_constraint_rhs(data::JuMPSolverData, cname)
    constr = data.cname_to_constr[cname]
    return get_constraint_rhs(constr)
end


function get_constraint_lhs(data::JuMPSolverData, cname)
    constr = data.cname_to_constr[cname]
    return get_constraint_lhs(constr)
end


function get_constraint_sense(data::JuMPSolverData, cname)
    constr = data.cname_to_constr[cname]
    return get_constraint_sense(constr)
end


# Constraints: ScalarAffineFunction, LessThan
# -------------------------------------------------------------------------
function get_constraint_rhs(
    constr::ConstraintRef{
        Model,
        MathOptInterface.ConstraintIndex{
            MathOptInterface.ScalarAffineFunction{T},
            MathOptInterface.LessThan{T},
        },
        ScalarShape,
    },
)::T where T
    return MOI.get(
        constr.model.moi_backend,
        MOI.ConstraintSet(),
        constr.index,
    ).upper
end


function _terms_dict(constr)
    terms = MOI.get(
        constr.model.moi_backend,
        MOI.ConstraintFunction(),
        constr.index,
    ).terms
    return Dict(
        MOI.get(
            constr.model.moi_backend,
            MOI.VariableName(),
            term.variable_index
        ) => term.coefficient
        for term in terms
    )
end


function get_constraint_lhs(
    constr::ConstraintRef{
        Model,
        MathOptInterface.ConstraintIndex{
            MathOptInterface.ScalarAffineFunction{T},
            MathOptInterface.LessThan{T},
        },
        ScalarShape,
    },
)::Dict{String, T} where T
    return _terms_dict(constr)
end


function get_constraint_sense(
    constr::ConstraintRef{
        Model,
        MathOptInterface.ConstraintIndex{
            MathOptInterface.ScalarAffineFunction{T},
            MathOptInterface.LessThan{T},
        },
        ScalarShape,
    },
)::String where T
    return "<"
end


# Constraints: ScalarAffineFunction, GreaterThan
# -------------------------------------------------------------------------
function get_constraint_rhs(
    constr::ConstraintRef{
        Model,
        MathOptInterface.ConstraintIndex{
            MathOptInterface.ScalarAffineFunction{T},
            MathOptInterface.GreaterThan{T},
        },
        ScalarShape,
    },
)::T where T
    return MOI.get(
        constr.model.moi_backend,
        MOI.ConstraintSet(),
        constr.index,
    ).lower
end


function get_constraint_lhs(
    constr::ConstraintRef{
        Model,
        MathOptInterface.ConstraintIndex{
            MathOptInterface.ScalarAffineFunction{T},
            MathOptInterface.GreaterThan{T},
        },
        ScalarShape,
    },
)::Dict{String, T} where T
    return _terms_dict(constr)
end


function get_constraint_sense(
    constr::ConstraintRef{
        Model,
        MathOptInterface.ConstraintIndex{
            MathOptInterface.ScalarAffineFunction{T},
            MathOptInterface.GreaterThan{T},
        },
        ScalarShape,
    },
)::String where T
    return ">"
end


# Constraints: ScalarAffineFunction, EqualTo
# -------------------------------------------------------------------------
function get_constraint_rhs(
    constr::ConstraintRef{
        Model,
        MathOptInterface.ConstraintIndex{
            MathOptInterface.ScalarAffineFunction{T},
            MathOptInterface.EqualTo{T},
        },
        ScalarShape,
    },
)::T where T
    return MOI.get(
        constr.model.moi_backend,
        MOI.ConstraintSet(),
        constr.index,
    ).value
end


function get_constraint_lhs(
    constr::ConstraintRef{
        Model,
        MathOptInterface.ConstraintIndex{
            MathOptInterface.ScalarAffineFunction{T},
            MathOptInterface.EqualTo{T},
        },
        ScalarShape,
    },
)::Dict{String, T} where T
    return _terms_dict(constr)
end


function get_constraint_sense(
    constr::ConstraintRef{
        Model,
        MathOptInterface.ConstraintIndex{
            MathOptInterface.ScalarAffineFunction{T},
            MathOptInterface.EqualTo{T},
        },
        ScalarShape,
    },
)::String where T
    return "="
end


@pydef mutable struct JuMPSolver <: miplearn.solvers.internal.InternalSolver
    function __init__(self; optimizer)
        self.data = JuMPSolverData(
            nothing,  # varname_to_var
            optimizer,
            nothing,  # instance
            nothing,  # model
            nothing,  # bin_vars
            nothing,  # solution
            nothing,  # cname_to_constr
        ) 
    end

    set_warm_start(self, solution) =
        set_warm_start!(self.data, solution)

    fix(self, solution) =
        fix!(self.data, solution)
    
    set_instance(self, instance, model) =
        set_instance!(self.data, instance, model)
    
    solve(
        self;
        tee=false,
        iteration_cb,
        lazy_cb,
        user_cut_cb,
    ) = solve(
        self.data,
        tee=tee,
        iteration_cb=iteration_cb,
    )
    
    solve_lp(self; tee=false) =
        solve_lp(self.data, tee=tee)
    
    get_solution(self) =
        self.data.solution

    get_variables(self) =
        get_variables(self.data)
    
    set_branching_priorities(self, priorities) =
        @warn "JuMPSolver: set_branching_priorities not implemented"
    
    add_constraint(self, constraint) =
        nothing

    get_variable_names(self) =
        get_variable_names(self.data)

    is_infeasible(self) =
        is_infeasible(self.data)

    get_constraint_ids(self) =
        get_constraint_ids(self.data)

    get_constraint_rhs(self, cname) =
        get_constraint_rhs(self.data, cname)

    get_constraint_lhs(self, cname) =
        get_constraint_lhs(self.data, cname)

    get_constraint_sense(self, cname) =
        get_constraint_sense(self.data, cname)

    clone(self) = self

    add_cut(self) = error("not implemented")
    extract_constraint(self) = error("not implemented")
    is_constraint_satisfied(self) = error("not implemented")
    set_constraint_sense(self) = error("not implemented")
    relax(self) = error("not implemented")
    get_inequality_slacks(self) = error("not implemented")
    get_dual(self) = error("not implemented")
    get_sense(self) = error("not implemented")
end

export JuMPSolver, solve!, fit!, add!
