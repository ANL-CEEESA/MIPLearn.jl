#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using SparseArrays

@inline frac(x::Float64) = x - floor(x)

function select_gmi_rows(data, basis, x; max_rows=10, atol=0.001)
    candidate_rows = [
        r
        for r in 1:length(basis.var_basic)
        if (data.var_types[basis.var_basic[r]] != 'C') && (frac(x[basis.var_basic[r]]) > atol)
    ]
    candidate_vals = frac.(x[basis.var_basic[candidate_rows]])
    score = abs.(candidate_vals .- 0.5)
    perm = sortperm(score)
    return [candidate_rows[perm[i]] for i in 1:min(length(perm), max_rows)]
end

function compute_gmi(
    data::ProblemData,
    tableau::Tableau,
    tol=1e-8,
)::ConstraintSet
    nrows, ncols = size(tableau.lhs)
    ub = Float64[Inf for _ in 1:nrows]
    lb = Float64[0.999 for _ in 1:nrows]
    tableau_I, tableau_J, tableau_V = findnz(tableau.lhs)
    lhs_I = Int[]
    lhs_J = Int[]
    lhs_V = Float64[]
    @timeit "Compute coefficients" begin
        for k in 1:nnz(tableau.lhs)
            i::Int = tableau_I[k]
            v::Float64 = 0.0
            alpha_j = frac(tableau_V[k])
            beta = frac(tableau.rhs[i])
            if data.var_types[i] == "C"
                if alpha_j >= 0
                    v = alpha_j / beta
                else
                    v = alpha_j / (1 - beta)
                end
            else
                if alpha_j <= beta
                    v = alpha_j / beta
                else
                    v = (1 - alpha_j) / (1 - beta)
                end
            end
            if abs(v) > tol
                push!(lhs_I, i)
                push!(lhs_J, tableau_J[k])
                push!(lhs_V, v)
            end
        end
        lhs = sparse(lhs_I, lhs_J, lhs_V, nrows, ncols)
    end
    return ConstraintSet(; lhs, ub, lb)
end

function assert_cuts_off(
    cuts::ConstraintSet,
    x::Vector{Float64},
    tol=1e-6
)
    for i in 1:length(cuts.lb)
        val = cuts.lhs[i, :]' * x
        if (val <= cuts.ub[i] - tol) && (val >= cuts.lb[i] + tol)
            throw(ErrorException("inequality fails to cut off fractional solution"))
        end
    end
end

function assert_does_not_cut_off(
    cuts::ConstraintSet,
    x::Vector{Float64};
    tol=1e-6
)
    for i in 1:length(cuts.lb)
        val = cuts.lhs[i, :]' * x
        ub = cuts.ub[i]
        lb = cuts.lb[i]
        if (val >= ub) || (val <= lb)
            throw(ErrorException("inequality $i cuts off integer solution ($lb <= $val <= $ub)"))
        end
    end
end

export compute_gmi, frac, select_gmi_rows, assert_cuts_off, assert_does_not_cut_off