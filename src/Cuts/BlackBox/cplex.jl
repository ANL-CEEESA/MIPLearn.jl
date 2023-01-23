#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using CPLEX
using JuMP
using HDF5

struct CplexBlackBoxCuts end

function collect(
    mps_filename::String,
    ::CplexBlackBoxCuts,
)::Nothing
    tempdir = mktempdir()
    isfile(mps_filename) || error("file not found: $mps_filename")
    h5_filename = replace(mps_filename, ".mps.gz" => ".h5")

    # Initialize CPLEX
    status_p = [Cint(0)]
    env = CPXopenCPLEX(status_p)

    # Parameter: Disable presolve
    CPXsetintparam(env, CPX_PARAM_AGGFILL, 0)
    CPXsetintparam(env, CPX_PARAM_AGGIND, 0)
    CPXsetintparam(env, CPX_PARAM_PREIND, 0)
    CPXsetintparam(env, CPX_PARAM_PREPASS, 0)
    CPXsetintparam(env, CPX_PARAM_REDUCE, 0)
    CPXsetintparam(env, CPX_PARAM_PREDUAL, -1)
    CPXsetintparam(env, CPX_PARAM_PRESLVND, -1)

    # Parameter: Enable logging
    CPXsetintparam(env, CPX_PARAM_SCRIND, 1)

    # Parameter: Stop processing at the root node
    CPXsetintparam(env, CPX_PARAM_NODELIM, 0)

    # Parameter: Make cutting plane generation more aggresive
    CPXsetintparam(env, CPX_PARAM_AGGCUTLIM, 100)
    CPXsetintparam(env, CPX_PARAM_FRACCAND, 1000)
    CPXsetintparam(env, CPX_PARAM_FRACCUTS, 2)
    CPXsetintparam(env, CPX_PARAM_FRACPASS, 100)
    CPXsetintparam(env, CPX_PARAM_GUBCOVERS, 100)
    CPXsetintparam(env, CPX_PARAM_MIRCUTS, 2)
    CPXsetintparam(env, CPX_PARAM_ZEROHALFCUTS, 2)

    # Load problem
    lp = CPXcreateprob(env, status_p, "problem")
    CPXreadcopyprob(env, lp, mps_filename, "mps")

    # Define callback
    function solve_callback(env, cbdata, wherefrom, cbhandle, useraction_p)::Int32
        nodelp_p = [CPXLPptr(0)]
        CPXgetcallbacknodelp(env, cbdata, wherefrom, nodelp_p)
        CPXwriteprob(env, nodelp_p[1], "$tempdir/root.mps", C_NULL)
        return 0
    end
    c_solve_callback = @cfunction($solve_callback, Cint, (
        CPXENVptr,  # env
        Ptr{Cvoid}, # cbdata
        Cint,       # wherefrom
        Ptr{Cvoid}, # cbhandle
        Ptr{Cint},  # useraction_p
    ))
    CPXsetsolvecallbackfunc(env, c_solve_callback, C_NULL)

    # Run optimization
    CPXmipopt(env, lp)

    # Load generated MPS file
    model = JuMP.read_from_file("$tempdir/root.mps")

    # Parse cuts
    cuts_lhs::Vector{Vector{Float64}} = []
    cuts_rhs::Vector{Float64} = []
    nvars = num_variables(model)
    constraints = all_constraints(model, GenericAffExpr{Float64,VariableRef}, MOI.LessThan{Float64})
    for conRef in constraints
        if name(conRef)[begin] in ['i', 'f', 'm', 'r', 'L', 'z', 'v'] &&
            isdigit(name(conRef)[begin+1])
            c = constraint_object(conRef)
            cset = MOI.get(conRef.model.moi_backend, MOI.ConstraintSet(), conRef.index)
            lhs = zeros(nvars)
            for (key, val) in c.func.terms
                lhs[key.index.value] = val
            end
            push!(cuts_lhs, lhs)
            push!(cuts_rhs, cset.upper)
        end
    end
    @info "$(length(cuts_lhs)) CPLEX cuts collected"
    cuts_lhs_matrix::Matrix{Float64} = vcat(cuts_lhs'...)

    # Store cuts in HDF5 file
    h5open(h5_filename, "r+") do h5
        for key in ["cuts_cpx_lhs", "cuts_cpx_rhs"]
            if haskey(h5, key)
                delete_object(h5, key)
            end
        end
        write(h5, "cuts_cpx_lhs", cuts_lhs_matrix)
        write(h5, "cuts_cpx_rhs", cuts_rhs)
    end

    return
end

export CplexBlackBoxCuts, collect
