#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2022, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using CPLEX

function _probe(
    mip::MIP,
    cpx::CPLEX.Optimizer,
    var::Variable,
    ::Float64,
    ::Float64,
    ::Float64,
    itlim::Int,
)::Tuple{Float64,Float64}
    indices = [var.index - Cint(1)]
    downobj, upobj, cnt = [0.0], [0.0], 1

    status = CPXlpopt(cpx.env, cpx.lp)
    status == 0 || error("CPXlpopt failed ($status)")

    status = CPXstrongbranch(
        cpx.env,
        cpx.lp,
        indices,
        cnt,
        downobj,
        upobj,
        itlim,
    )
    status == 0 || error("CPXstrongbranch failed ($status)")

    return upobj[1] * mip.sense, downobj[1] * mip.sense
end


function _relax_integrality!(cpx::CPLEX.Optimizer)::Nothing
    status = CPXchgprobtype(cpx.env, cpx.lp, CPLEX.CPXPROB_LP)
    status == 0 || error("CPXchgprobtype failed ($status)")
    return
end