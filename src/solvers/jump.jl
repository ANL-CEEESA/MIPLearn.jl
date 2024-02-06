#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using HiGHS
using JSON

global JumpModel = PyNULL()

Base.@kwdef mutable struct _JumpModelExtData
    aot_cuts = nothing
    cb_data = nothing
    cuts = []
    lazy = []
    where::Symbol = :WHERE_DEFAULT
    cuts_enforce::Union{Function,Nothing} = nothing
    cuts_separate::Union{Function,Nothing} = nothing
    lazy_enforce::Union{Function,Nothing} = nothing
    lazy_separate::Union{Function,Nothing} = nothing
    lp_optimizer
end

function JuMP.copy_extension_data(
    old_ext::_JumpModelExtData,
    new_model::AbstractModel,
    ::AbstractModel,
)
    new_model.ext[:miplearn] = _JumpModelExtData(
        lp_optimizer=old_ext.lp_optimizer
    )
end

# -----------------------------------------------------------------------------

function _add_constrs(
    model::JuMP.Model,
    var_names,
    constrs_lhs,
    constrs_sense,
    constrs_rhs,
    stats,
)
    n, m = length(var_names), length(constrs_rhs)
    vars = [variable_by_name(model, v) for v in var_names]
    for i = 1:m
        lhs = sum(constrs_lhs[i, j] * vars[j] for j = 1:n)
        sense = constrs_sense[i]
        rhs = constrs_rhs[i]
        if sense == "="
            @constraint(model, lhs == rhs)
        elseif sense == ">"
            @constraint(model, lhs >= rhs)
        elseif sense == "<"
            @constraint(model, lhs <= rhs)
        else
            error("Unknown sense: $sense")
        end
    end
end

function submit(model::JuMP.Model, constr)
    ext = model.ext[:miplearn]
    if ext.where == :WHERE_CUTS
        MOI.submit(model, MOI.UserCut(ext.cb_data), constr)
    elseif ext.where == :WHERE_LAZY
        MOI.submit(model, MOI.LazyConstraint(ext.cb_data), constr)
    else
        add_constraint(model, constr)
    end
end

function _extract_after_load(model::JuMP.Model, h5)
    if JuMP.objective_sense(model) == MOI.MIN_SENSE
        h5.put_scalar("static_sense", "min")
    else
        h5.put_scalar("static_sense", "max")
    end
    _extract_after_load_vars(model, h5)
    _extract_after_load_constrs(model, h5)
end

function _extract_after_load_vars(model::JuMP.Model, h5)
    vars = JuMP.all_variables(model)
    lb = [
        JuMP.is_binary(v) ? 0.0 : JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : -Inf
        for v in vars
    ]
    ub = [
        JuMP.is_binary(v) ? 1.0 : JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : Inf
        for v in vars
    ]
    types = [JuMP.is_binary(v) ? "B" : JuMP.is_integer(v) ? "I" : "C" for v in vars]
    obj = objective_function(model, AffExpr)
    obj_coeffs = [v ∈ keys(obj.terms) ? obj.terms[v] : 0.0 for v in vars]
    h5.put_array("static_var_names", to_str_array(JuMP.name.(vars)))
    h5.put_array("static_var_types", to_str_array(types))
    h5.put_array("static_var_lower_bounds", lb)
    h5.put_array("static_var_upper_bounds", ub)
    h5.put_array("static_var_obj_coeffs", obj_coeffs)
    h5.put_scalar("static_obj_offset", obj.constant)
end

function _extract_after_load_constrs(model::JuMP.Model, h5)
    names = String[]
    senses, rhs = String[], Float64[]
    lhs_rows, lhs_cols, lhs_values = Int[], Int[], Float64[]

    constr_index = 1
    for (ftype, stype) in JuMP.list_of_constraint_types(model)
        for constr in JuMP.all_constraints(model, ftype, stype)
            cset = MOI.get(constr.model.moi_backend, MOI.ConstraintSet(), constr.index)
            cf = MOI.get(constr.model.moi_backend, MOI.ConstraintFunction(), constr.index)

            name = JuMP.name(constr)
            length(name) > 0 || continue
            push!(names, name)

            # LHS, RHS and sense
            if ftype == VariableRef
                # nop
            elseif ftype == AffExpr
                if stype == MOI.EqualTo{Float64}
                    rhs_c = cset.value
                    push!(senses, "=")
                elseif stype == MOI.LessThan{Float64}
                    rhs_c = cset.upper
                    push!(senses, "<")
                elseif stype == MOI.GreaterThan{Float64}
                    rhs_c = cset.lower
                    push!(senses, ">")
                else
                    error("Unsupported set: $stype")
                end
                push!(rhs, rhs_c)
                for term in cf.terms
                    push!(lhs_cols, term.variable.value)
                    push!(lhs_rows, constr_index)
                    push!(lhs_values, term.coefficient)
                end
                constr_index += 1
            else
                error("Unsupported constraint type: ($ftype, $stype)")
            end
        end
    end
    if isempty(names)
        error("no model constraints found; note that MIPLearn ignores unnamed constraints")
    end
    lhs = sparse(lhs_rows, lhs_cols, lhs_values, length(rhs), JuMP.num_variables(model))
    h5.put_sparse("static_constr_lhs", lhs)
    h5.put_array("static_constr_rhs", rhs)
    h5.put_array("static_constr_sense", to_str_array(senses))
    h5.put_array("static_constr_names", to_str_array(names))
end

function _extract_after_lp(model::JuMP.Model, h5)
    h5.put_scalar("lp_wallclock_time", solve_time(model))
    h5.put_scalar("lp_obj_value", objective_value(model))
    _extract_after_lp_vars(model, h5)
    _extract_after_lp_constrs(model, h5)
end

function _extract_after_lp_vars(model::JuMP.Model, h5)
    # Values and reduced costs
    vars = all_variables(model)
    h5.put_array("lp_var_values", JuMP.value.(vars))
    h5.put_array("lp_var_reduced_costs", reduced_cost.(vars))

    # Basis status
    basis_status = []
    for var in vars
        bstatus = MOI.get(model, MOI.VariableBasisStatus(), var)
        if bstatus == MOI.BASIC
            bstatus_v = "B"
        elseif bstatus == MOI.NONBASIC_AT_LOWER
            bstatus_v = "L"
        elseif bstatus == MOI.NONBASIC_AT_UPPER
            bstatus_v = "U"
        else
            error("Unknown basis status: $(bstatus)")
        end
        push!(basis_status, bstatus_v)
    end
    h5.put_array("lp_var_basis_status", to_str_array(basis_status))

    # Sensitivity analysis
    obj_coeffs = h5.get_array("static_var_obj_coeffs")
    sensitivity_report = lp_sensitivity_report(model)
    sa_obj_down, sa_obj_up = Float64[], Float64[]
    sa_lb_down, sa_lb_up = Float64[], Float64[]
    sa_ub_down, sa_ub_up = Float64[], Float64[]
    for (i, v) in enumerate(vars)
        # Objective function
        (delta_down, delta_up) = sensitivity_report[v]
        push!(sa_obj_down, delta_down + obj_coeffs[i])
        push!(sa_obj_up, delta_up + obj_coeffs[i])

        # Lower bound
        if has_lower_bound(v)
            constr = LowerBoundRef(v)
            (delta_down, delta_up) = sensitivity_report[constr]
            push!(sa_lb_down, lower_bound(v) + delta_down)
            push!(sa_lb_up, lower_bound(v) + delta_up)
        else
            push!(sa_lb_down, -Inf)
            push!(sa_lb_up, -Inf)
        end

        # Upper bound
        if has_upper_bound(v)
            constr = JuMP.UpperBoundRef(v)
            (delta_down, delta_up) = sensitivity_report[constr]
            push!(sa_ub_down, upper_bound(v) + delta_down)
            push!(sa_ub_up, upper_bound(v) + delta_up)
        else
            push!(sa_ub_down, Inf)
            push!(sa_ub_up, Inf)
        end
    end
    h5.put_array("lp_var_sa_obj_up", sa_obj_up)
    h5.put_array("lp_var_sa_obj_down", sa_obj_down)
    h5.put_array("lp_var_sa_ub_up", sa_ub_up)
    h5.put_array("lp_var_sa_ub_down", sa_ub_down)
    h5.put_array("lp_var_sa_lb_up", sa_lb_up)
    h5.put_array("lp_var_sa_lb_down", sa_lb_down)
end


function _extract_after_lp_constrs(model::JuMP.Model, h5)
    # Slacks
    lhs = h5.get_sparse("static_constr_lhs")
    rhs = h5.get_array("static_constr_rhs")
    x = h5.get_array("lp_var_values")
    slacks = abs.(lhs * x - rhs)
    h5.put_array("lp_constr_slacks", slacks)

    sa_rhs_up, sa_rhs_down = Float64[], Float64[]
    duals = Float64[]
    basis_status = []
    constr_idx = 1
    sensitivity_report = lp_sensitivity_report(model)
    for (ftype, stype) in JuMP.list_of_constraint_types(model)
        for constr in JuMP.all_constraints(model, ftype, stype)
            length(JuMP.name(constr)) > 0 || continue

            # Duals
            push!(duals, JuMP.dual(constr))

            # Basis status
            b = MOI.get(model, MOI.ConstraintBasisStatus(), constr)
            if b == MOI.NONBASIC
                push!(basis_status, "N")
            elseif b == MOI.BASIC
                push!(basis_status, "B")
            else
                error("Unknown basis status: $b")
            end

            # Sensitivity analysis
            (delta_down, delta_up) = sensitivity_report[constr]
            push!(sa_rhs_down, rhs[constr_idx] + delta_down)
            push!(sa_rhs_up, rhs[constr_idx] + delta_up)

            constr_idx += 1
        end
    end
    h5.put_array("lp_constr_dual_values", duals)
    h5.put_array("lp_constr_basis_status", to_str_array(basis_status))
    h5.put_array("lp_constr_sa_rhs_up", sa_rhs_up)
    h5.put_array("lp_constr_sa_rhs_down", sa_rhs_down)
end

function _extract_after_mip(model::JuMP.Model, h5)
    h5.put_scalar("mip_obj_value", objective_value(model))
    h5.put_scalar("mip_obj_bound", objective_bound(model))
    h5.put_scalar("mip_wallclock_time", solve_time(model))
    h5.put_scalar("mip_gap", relative_gap(model))

    # Values
    vars = all_variables(model)
    x = JuMP.value.(vars)
    h5.put_array("mip_var_values", x)

    # Slacks
    lhs = h5.get_sparse("static_constr_lhs")
    rhs = h5.get_array("static_constr_rhs")
    slacks = abs.(lhs * x - rhs)
    h5.put_array("mip_constr_slacks", slacks)

    # Cuts and lazy constraints
    ext = model.ext[:miplearn]
    h5.put_scalar("mip_cuts", JSON.json(ext.cuts))
    h5.put_scalar("mip_lazy", JSON.json(ext.lazy))
end

function _fix_variables(model::JuMP.Model, var_names, var_values, stats)
    vars = [variable_by_name(model, v) for v in var_names]
    for (i, var) in enumerate(vars)
        fix(var, var_values[i], force=true)
    end
end

function _optimize(model::JuMP.Model)
    # Set up cut callbacks
    ext = model.ext[:miplearn]
    ext.cuts = []
    function cut_callback(cb_data)
        ext.cb_data = cb_data
        ext.where = :WHERE_CUTS
        if ext.aot_cuts !== nothing
            @info "Enforcing $(length(ext.aot_cuts)) cuts ahead-of-time..."
            violations = ext.aot_cuts
            ext.aot_cuts = nothing
        else
            violations = ext.cuts_separate(cb_data)
            for v in violations
                push!(ext.cuts, v)
            end
        end
        if !isempty(violations)
            ext.cuts_enforce(violations)
        end
    end
    if ext.cuts_separate !== nothing
        set_attribute(model, MOI.UserCutCallback(), cut_callback)
    end

    # Set up lazy constraint callbacks
    ext.lazy = []
    function lazy_callback(cb_data)
        ext.cb_data = cb_data
        ext.where = :WHERE_LAZY
        violations = ext.lazy_separate(cb_data)
        for v in violations
            push!(ext.lazy, v)
        end
        if !isempty(violations)
            ext.lazy_enforce(violations)
        end
    end
    if ext.lazy_separate !== nothing
        set_attribute(model, MOI.LazyConstraintCallback(), lazy_callback)
    end

    # Optimize
    optimize!(model)

    # Cleanup
    ext.where = :WHERE_DEFAULT
    ext.cb_data = nothing
    flush(stdout)
    Libc.flush_cstdio()
end

function _relax(model::JuMP.Model)
    relaxed, _ = copy_model(model)
    relax_integrality(relaxed)
    set_optimizer(relaxed, model.ext[:miplearn].lp_optimizer)
    set_silent(relaxed)
    return relaxed
end

function _set_warm_starts(model::JuMP.Model, var_names, var_values, stats)
    (n_starts, _) = size(var_values)
    n_starts == 1 || error("JuMP does not support multiple warm starts")
    vars = [variable_by_name(model, v) for v in var_names]
    for (i, var) in enumerate(vars)
        if isfinite(var_values[i])
            set_start_value(var, var_values[i])
        end
    end
end

function _write(model::JuMP.Model, filename)
    ext = model.ext[:miplearn]
    if ext.lazy_separate !== nothing
        set_attribute(model, MOI.LazyConstraintCallback(), nothing)
    end
    if ext.cuts_separate !== nothing
        set_attribute(model, MOI.UserCutCallback(), nothing)
    end
    write_to_file(model, filename)
end

# -----------------------------------------------------------------------------

function __init_solvers_jump__()
    AbstractModel = pyimport("miplearn.solvers.abstract").AbstractModel
    @pydef mutable struct Class <: AbstractModel

        function __init__(
            self,
            inner;
            cuts_enforce::Union{Function,Nothing}=nothing,
            cuts_separate::Union{Function,Nothing}=nothing,
            lazy_enforce::Union{Function,Nothing}=nothing,
            lazy_separate::Union{Function,Nothing}=nothing,
            lp_optimizer=HiGHS.Optimizer,
        )
            self.inner = inner
            self.inner.ext[:miplearn] = _JumpModelExtData(
                cuts_enforce=cuts_enforce,
                cuts_separate=cuts_separate,
                lazy_enforce=lazy_enforce,
                lazy_separate=lazy_separate,
                lp_optimizer=lp_optimizer,
            )
        end

        add_constrs(
            self,
            var_names,
            constrs_lhs,
            constrs_sense,
            constrs_rhs,
            stats=nothing,
        ) = _add_constrs(
            self.inner,
            from_str_array(var_names),
            constrs_lhs,
            from_str_array(constrs_sense),
            constrs_rhs,
            stats,
        )

        extract_after_load(self, h5) = _extract_after_load(self.inner, h5)

        extract_after_lp(self, h5) = _extract_after_lp(self.inner, h5)

        extract_after_mip(self, h5) = _extract_after_mip(self.inner, h5)

        fix_variables(self, var_names, var_values, stats=nothing) =
            _fix_variables(self.inner, from_str_array(var_names), var_values, stats)

        optimize(self) = _optimize(self.inner)

        relax(self) = Class(_relax(self.inner))

        set_warm_starts(self, var_names, var_values, stats=nothing) =
            _set_warm_starts(self.inner, from_str_array(var_names), var_values, stats)

        write(self, filename) = _write(self.inner, filename)

        function set_cuts(self, cuts)
            self.inner.ext[:miplearn].aot_cuts = cuts
        end

        function lazy_enforce(self, violations)
            self.inner.ext[:miplearn].lazy_enforce(violations)
        end

        function _lazy_enforce_collected(self)
            ext = self.inner.ext[:miplearn]
            if ext.lazy_enforce !== nothing
                ext.lazy_enforce(ext.lazy)
            end
        end
    end
    copy!(JumpModel, Class)
end

export JumpModel
