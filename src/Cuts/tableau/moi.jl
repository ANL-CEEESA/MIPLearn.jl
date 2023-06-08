#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP

function ProblemData(model::Model)::ProblemData
    vars = all_variables(model)

    # Objective function
    obj = objective_function(model)
    obj = [v ∈ keys(obj.terms) ? obj.terms[v] : 0.0 for v in vars]

    # Variable types, lower bounds and upper bounds
    var_lb = [is_binary(v) ? 0.0 : has_lower_bound(v) ? lower_bound(v) : -Inf for v in vars]
    var_ub = [is_binary(v) ? 1.0 : has_upper_bound(v) ? upper_bound(v) : Inf for v in vars]
    var_types = [is_binary(v) || is_integer(v) ? 'I' : 'C' for v in vars]
    var_names = [name(v) for v in vars]

    # Constraints
    constr_lb = Float64[]
    constr_ub = Float64[]
    constr_lhs_rows = Int[]
    constr_lhs_cols = Int[]
    constr_lhs_values = Float64[]
    constr_index = 1
    for (ftype, stype) in list_of_constraint_types(model)
        for constr in all_constraints(model, ftype, stype)
            cset = MOI.get(constr.model.moi_backend, MOI.ConstraintSet(), constr.index)
            cf = MOI.get(constr.model.moi_backend, MOI.ConstraintFunction(), constr.index)
            if ftype == VariableRef
                var_idx = cf.value
                if stype == MOI.Integer || stype == MOI.ZeroOne
                    # nop
                elseif stype == MOI.EqualTo{Float64}
                    var_lb[var_idx] = max(var_lb[var_idx], cset.value)
                    var_ub[var_idx] = min(var_ub[var_idx], cset.value)
                elseif stype == MOI.LessThan{Float64}
                    var_ub[var_idx] = min(var_ub[var_idx], cset.upper)
                elseif stype == MOI.GreaterThan{Float64}
                    var_lb[var_idx] = max(var_lb[var_idx], cset.lower)
                elseif stype == MOI.Interval{Float64}
                    var_lb[var_idx] = max(var_lb[var_idx], cset.lower)
                    var_ub[var_idx] = min(var_ub[var_idx], cset.upper)
                else
                    error("Unsupported set: $stype")
                end
            elseif ftype == AffExpr
                if stype == MOI.EqualTo{Float64}
                    push!(constr_lb, cset.value)
                    push!(constr_ub, cset.value)
                elseif stype == MOI.LessThan{Float64}
                    push!(constr_lb, -Inf)
                    push!(constr_ub, cset.upper)
                elseif stype == MOI.GreaterThan{Float64}
                    push!(constr_lb, cset.lower)
                    push!(constr_ub, Inf)
                elseif stype == MOI.Interval{Float64}
                    push!(constr_lb, cset.lower)
                    push!(constr_ub, cset.upper)
                else
                    error("Unsupported set: $stype")
                end
                for term in cf.terms
                    push!(constr_lhs_cols, term.variable.value)
                    push!(constr_lhs_rows, constr_index)
                    push!(constr_lhs_values, term.coefficient)
                end
                constr_index += 1
            else
                error("Unsupported constraint type: ($ftype, $stype)")
            end
        end
    end

    n = length(vars)
    m = constr_index - 1
    constr_lhs = sparse(constr_lhs_rows, constr_lhs_cols, constr_lhs_values, m, n)

    @assert length(obj) == n
    @assert length(var_lb) == n
    @assert length(var_ub) == n
    @assert length(var_types) == n
    @assert length(var_names) == n
    @assert length(constr_lb) == m
    @assert length(constr_ub) == m

    return ProblemData(
        obj_offset = 0.0;
        obj,
        constr_lb,
        constr_ub,
        constr_lhs,
        var_lb,
        var_ub,
        var_types,
        var_names,
    )
end

function to_model(data::ProblemData, tol = 1e-6)::Model
    model = Model()

    # Variables
    nvars = length(data.obj)
    @variable(model, x[1:nvars])
    for i = 1:nvars
        set_name(x[i], data.var_names[i])
        if data.var_types[i] == 'B'
            set_binary(x[i])
        elseif data.var_types[i] == 'I'
            set_integer(x[i])
        end
        if isfinite(data.var_lb[i])
            set_lower_bound(x[i], data.var_lb[i])
        end
        if isfinite(data.var_ub[i])
            set_upper_bound(x[i], data.var_ub[i])
        end
        set_objective_coefficient(model, x[i], data.obj[i])
    end

    # Constraints
    lhs = data.constr_lhs * x
    for (j, lhs_expr) in enumerate(lhs)
        lb = data.constr_lb[j]
        ub = data.constr_ub[j]
        if abs(lb - ub) < tol
            @constraint(model, lb == lhs_expr)
        elseif isfinite(lb) && !isfinite(ub)
            @constraint(model, lb <= lhs_expr)
        elseif !isfinite(lb) && isfinite(ub)
            @constraint(model, lhs_expr <= ub)
        else
            @constraint(model, lb <= lhs_expr <= ub)
        end
    end

    return model
end

function add_constraint_set(model::JuMP.Model, cs::ConstraintSet)
    vars = all_variables(model)
    nrows, _ = size(cs.lhs)
    constrs = []
    for i = 1:nrows
        c = nothing
        if isinf(cs.ub[i])
            c = @constraint(model, cs.lb[i] <= dot(cs.lhs[i, :], vars))
        elseif isinf(cs.lb[i])
            c = @constraint(model, dot(cs.lhs[i, :], vars) <= cs.ub[i])
        else
            c = @constraint(model, cs.lb[i] <= dot(cs.lhs[i, :], vars) <= cs.ub[i])
        end
        push!(constrs, c)
    end
    return constrs
end

function set_warm_start(model::JuMP.Model, x::Vector{Float64})
    vars = all_variables(model)
    for (i, xi) in enumerate(x)
        set_start_value(vars[i], xi)
    end
end

export to_model, ProblemData, add_constraint_set, set_warm_start
