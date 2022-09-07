#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Clp
using JuMP
using Test
using MIPLearn.BB

basepath = @__DIR__

function runtests(optimizer_name, optimizer; large = true)
    @testset "Solve ($optimizer_name)" begin
        @testset "interface" begin
            filename = "$basepath/../fixtures/danoint.mps.gz"

            mip = BB.init(optimizer)
            BB.read!(mip, filename)

            @test mip.sense == 1.0
            @test length(mip.int_vars) == 56

            status, obj = BB.solve_relaxation!(mip)
            @test status == :Optimal
            @test round(obj, digits = 6) == 62.637280

            @test BB.name(mip, mip.int_vars[1]) == "xab"
            @test BB.name(mip, mip.int_vars[2]) == "xac"
            @test BB.name(mip, mip.int_vars[3]) == "xad"
            
            @test mip.int_vars_lb[1] == 0.0
            @test mip.int_vars_ub[1] == 1.0

            vals = BB.values(mip, mip.int_vars)
            @test round(vals[1], digits = 6) == 0.046933
            @test round(vals[2], digits = 6) == 0.000841
            @test round(vals[3], digits = 6) == 0.248696

            # Probe (up and down are feasible)
            probe_up, probe_down = BB.probe(mip, mip.int_vars[1], 0.5, 0.0, 1.0)
            @test round(probe_down, digits = 6) == 62.690000
            @test round(probe_up, digits = 6) == 62.714100

            # Fix one variable to zero
            BB.set_bounds!(mip, mip.int_vars[1:1], [0.0], [0.0])
            status, obj = BB.solve_relaxation!(mip)
            @test status == :Optimal
            @test round(obj, digits = 6) == 62.690000

            # Fix one variable to one and another variable variable to zero
            BB.set_bounds!(mip, mip.int_vars[1:2], [1.0, 0.0], [1.0, 0.0])
            status, obj = BB.solve_relaxation!(mip)
            @test status == :Optimal
            @test round(obj, digits = 6) == 62.714777

            # Probe (up is infeasible, down is feasible)
            BB.set_bounds!(mip, mip.int_vars[1:3], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0])
            status, obj = BB.solve_relaxation!(mip)
            @test status == :Optimal
            probe_up, probe_down = BB.probe(mip, mip.int_vars[3], 0.5, 0.0, 1.0)
            @test round(probe_up, digits = 6) == Inf
            @test round(probe_down, digits = 6) == 63.073992

            # Fix all binary variables to one, making problem infeasible
            N = length(mip.int_vars)
            BB.set_bounds!(mip, mip.int_vars, ones(N), ones(N))
            status, obj = BB.solve_relaxation!(mip)
            @test status == :Infeasible
            @test obj == Inf

            # Restore original problem
            N = length(mip.int_vars)
            BB.set_bounds!(mip, mip.int_vars, zeros(N), ones(N))
            status, obj = BB.solve_relaxation!(mip)
            @test status == :Optimal
            @test round(obj, digits = 6) == 62.637280
        end

        @testset "varbranch" begin
            branch_rules = [
                BB.RandomBranching(),
                BB.FirstInfeasibleBranching(),
                BB.LeastInfeasibleBranching(),
                BB.MostInfeasibleBranching(),
                BB.PseudocostBranching(),
                BB.StrongBranching(),
                BB.ReliabilityBranching(),
                BB.HybridBranching(),
            ]
            for branch_rule in branch_rules
                filename = "$basepath/../fixtures/vpm2.mps.gz"
                mip = BB.init(optimizer)
                BB.read!(mip, filename)
                @info optimizer_name, branch_rule
                @time BB.solve!(
                    mip,
                    initial_primal_bound = 13.75,
                    print_interval = 10,
                    node_limit = 100,
                    branch_rule = branch_rule,
                )
            end
        end
    end
end

@testset "BB" begin
    @time runtests("Clp", Clp.Optimizer)
    if is_gurobi_available
        using Gurobi
        @time runtests("Gurobi", Gurobi.Optimizer)
    end
end
