#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using SparseArrays
using TimerOutputs

function select_gmi_rows(data, basis, x; max_rows=10, atol=1e-4)
    candidate_rows = [
        r for
        r in 1:length(basis.var_basic) if (
            (data.var_types[basis.var_basic[r]] != 'C') &&
            (frac(x[basis.var_basic[r]]) > atol) &&
            (frac2(x[basis.var_basic[r]]) > atol)
        )
    ]
    candidate_vals = frac.(x[basis.var_basic[candidate_rows]])
    score = abs.(candidate_vals .- 0.5)
    perm = sortperm(score)
    return [candidate_rows[perm[i]] for i = 1:min(length(perm), max_rows)]
end

function compute_gmi(data::ProblemData, tableau::Tableau)::ConstraintSet
    nrows, ncols = size(tableau.lhs)
    ub = Float64[Inf for _ = 1:nrows]
    lb = Float64[0.9999 for _ = 1:nrows]
    tableau_I, tableau_J, tableau_V = findnz(tableau.lhs)
    lhs_I = Int[]
    lhs_J = Int[]
    lhs_V = Float64[]
    @timeit "Compute coefficients" begin
        for k in 1:nnz(tableau.lhs)
            i::Int = tableau_I[k]
            j::Int = tableau_J[k]
            v::Float64 = 0.0
            frac_alpha_j = frac(tableau_V[k])
            alpha_j = tableau_V[k]
            beta = frac(tableau.rhs[i])
            if data.var_types[j] == 'C'
                if alpha_j >= 0
                    v = alpha_j / beta
                else
                    v = -alpha_j / (1 - beta)
                end
            else
                if frac_alpha_j < beta
                    v = frac_alpha_j / beta
                else
                    v = (1 - frac_alpha_j) / (1 - beta)
                end
            end
            if abs(v) > 1e-8
                push!(lhs_I, i)
                push!(lhs_J, tableau_J[k])
                push!(lhs_V, v)
            end
        end
        lhs = sparse(lhs_I, lhs_J, lhs_V, nrows, ncols)
    end
    return ConstraintSet(; lhs, ub, lb)
end

export compute_gmi, frac, select_gmi_rows, assert_cuts_off, assert_does_not_cut_off
