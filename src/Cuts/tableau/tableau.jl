#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using KLU
using TimerOutputs

function get_basis(model::JuMP.Model)::Basis
    var_basic = Int[]
    var_nonbasic = Int[]
    constr_basic = Int[]
    constr_nonbasic = Int[]

    # Variables
    for (i, var) in enumerate(all_variables(model))
        bstatus = MOI.get(model, MOI.VariableBasisStatus(), var)
        if bstatus == MOI.BASIC
            push!(var_basic, i)
        elseif bstatus == MOI.NONBASIC_AT_LOWER
            push!(var_nonbasic, i)
        else
            error("Unknown basis status: $bstatus")
        end
    end

    # Constraints
    constr_index = 1
    for (ftype, stype) in list_of_constraint_types(model)
        for constr in all_constraints(model, ftype, stype)
            if ftype == VariableRef
                # nop
            elseif ftype == AffExpr
                bstatus = MOI.get(model, MOI.ConstraintBasisStatus(), constr)
                if bstatus == MOI.BASIC
                    push!(constr_basic, constr_index)
                elseif bstatus == MOI.NONBASIC
                    push!(constr_nonbasic, constr_index)
                else
                    error("Unknown basis status: $bstatus")
                end
                constr_index += 1
            else
                error("Unsupported constraint type: ($ftype, $stype)")
            end
        end
    end

    return Basis(; var_basic, var_nonbasic, constr_basic, constr_nonbasic)
end

function get_x(model::JuMP.Model)
    return JuMP.value.(all_variables(model))
end

function compute_tableau(
    data::ProblemData,
    basis::Basis,
    x::Vector{Float64};
    rows::Union{Vector{Int},Nothing}=nothing,
    tol=1e-8
)::Tableau
    @timeit "Split data" begin
        nrows, ncols = size(data.constr_lhs)
        lhs_slacks = sparse(I, nrows, nrows)
        lhs_b = [data.constr_lhs[:, basis.var_basic] lhs_slacks[:, basis.constr_basic]]
        obj_b = [data.obj[basis.var_basic]; zeros(length(basis.constr_basic))]
        if rows === nothing
            rows = 1:nrows
        end
    end

    @timeit "Factorize basis matrix" begin
        factor = klu(sparse(lhs_b'))
    end

    @timeit "Compute tableau LHS" begin
        tableau_lhs_I = Int[]
        tableau_lhs_J = Int[]
        tableau_lhs_V = Float64[]
        for k in 1:length(rows)
            @timeit "Prepare inputs" begin
                i = rows[k]
                e = zeros(nrows)
                e[i] = 1.0
            end
            @timeit "Solve" begin
                sol = factor \ e
            end
            @timeit "Multiply" begin
                row = sol' * data.constr_lhs
            end
            @timeit "Sparsify & copy" begin
                for (j, v) in enumerate(row)
                    if abs(v) < tol
                        continue
                    end
                    push!(tableau_lhs_I, k)
                    push!(tableau_lhs_J, j)
                    push!(tableau_lhs_V, v)
                end
            end
        end
        tableau_lhs = sparse(
            tableau_lhs_I,
            tableau_lhs_J,
            tableau_lhs_V,
            length(rows),
            ncols,
        )
    end

    @timeit "Compute tableau RHS" begin
        tableau_rhs = [x[basis.var_basic]; zeros(length(basis.constr_basic))][rows]
    end

    @timeit "Compute tableau objective row" begin
        sol = factor \ obj_b
        tableau_obj = -data.obj' + sol' * data.constr_lhs
        tableau_obj[abs.(tableau_obj).<tol] .= 0
    end

    return Tableau(
        obj=tableau_obj,
        lhs=tableau_lhs,
        rhs=tableau_rhs,
        z=dot(data.obj, x),
    )
end

export get_basis, get_x, compute_tableau
