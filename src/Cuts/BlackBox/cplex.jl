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
    CPXsetintparam(env, CPX_PARAM_FRACCUTS, 2)
    CPXsetintparam(env, CPX_PARAM_MIRCUTS, 2)
    CPXsetintparam(env, CPX_PARAM_ZEROHALFCUTS, 2)
    # CPXsetintparam(env, CPX_PARAM_AGGCUTLIM, 100)
    # CPXsetintparam(env, CPX_PARAM_FRACCAND, 1000)
    # CPXsetintparam(env, CPX_PARAM_FRACPASS, 100)
    # CPXsetintparam(env, CPX_PARAM_GUBCOVERS, 100)

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

    function select(cr)
        return name(cr)[begin] in ['i', 'f', 'm', 'r', 'L', 'z', 'v'] &&  isdigit(name(cr)[begin+1])
    end

    # Parse cuts
    constraints = all_constraints(model, GenericAffExpr{Float64,VariableRef}, MOI.LessThan{Float64})
    nvars = num_variables(model)
    ncuts = length([cr for cr in constraints if select(cr)])
    cuts_lhs = spzeros(ncuts, nvars)
    cuts_rhs = Float64[]
    cuts_var_names = String[]

    for i in 1:nvars
        push!(cuts_var_names, name(VariableRef(model, MOI.VariableIndex(i))))
    end

    offset = 1
    for conRef in constraints
        if select(conRef)
            c = constraint_object(conRef)
            cset = MOI.get(conRef.model.moi_backend, MOI.ConstraintSet(), conRef.index)
            for (key, val) in c.func.terms
                idx = key.index.value
                if (idx < 1 || idx > nvars)
                    error("invalid index: $idx")
                end
                cuts_lhs[offset, idx - 1] = val
            end
            push!(cuts_rhs, cset.upper)
            offset += 1
        end
    end
    
    @info "Storing $(length(cuts_rhs)) CPLEX cuts..."
    h5 = Hdf5Sample(h5_filename)
    h5.put_sparse("cuts_cpx_lhs", cuts_lhs)
    h5.put_array("cuts_cpx_rhs", cuts_rhs)
    h5.put_array("cuts_cpx_var_names", to_str_array(cuts_var_names))
    h5.file.close()

    return
end

export CplexBlackBoxCuts, collect
