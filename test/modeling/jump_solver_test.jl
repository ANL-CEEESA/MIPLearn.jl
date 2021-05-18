#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Test
using MIPLearn
using Gurobi
using PyCall
using JuMP

miplearn_tests = pyimport("miplearn.solvers.tests")
traceback = pyimport("traceback")

@testset "JuMPSolver" begin
    solver = JuMPSolver(Gurobi.Optimizer)
    try  
        miplearn_tests.run_internal_solver_tests(solver)
    catch e
        if isa(e, PyCall.PyError)
            printstyled("Uncaught Python exception:\n", bold=true, color=:red)
            traceback.print_exception(e.T, e.val, e.traceback)
        end
        rethrow()
    end
end
