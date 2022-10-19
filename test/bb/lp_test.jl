#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Clp
using JuMP
using Test
using MIPLearn.BB
using MIPLearn

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
            probe_up, probe_down = BB.probe(mip, mip.int_vars[1], 0.5, 0.0, 1.0, 1_000_000)
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
                BB.StrongBranching(aggregation=:min),
                BB.ReliabilityBranching(aggregation=:min),
            ]
            for branch_rule in branch_rules
                for instance in ["bell5", "vpm2"]
                    h5 = Hdf5Sample("$basepath/../fixtures/$instance.h5")
                    mip_lower_bound = h5.get_scalar("mip_lower_bound")
                    mip_upper_bound = h5.get_scalar("mip_upper_bound")
                    mip_sense = h5.get_scalar("mip_sense")
                    mip_primal_bound = mip_sense == "min" ? mip_upper_bound : mip_lower_bound
                    h5.file.close()

                    mip = BB.init(optimizer)
                    BB.read!(mip, "$basepath/../fixtures/$instance.mps.gz")
                    @info optimizer_name, branch_rule, instance
                    @time BB.solve!(
                        mip,
                        initial_primal_bound = mip_primal_bound,
                        print_interval = 10,
                        node_limit = 100,
                        branch_rule = branch_rule,
                    )
                end
            end
        end
    end
end

@testset "BB" begin
    # @time runtests(
    #     "Clp",
    #     optimizer_with_attributes(
    #         Clp.Optimizer,
    #     ),
    # )

    if is_gurobi_available
        using Gurobi
        @time runtests(
            "Gurobi",
            optimizer_with_attributes(
                Gurobi.Optimizer,
                "Threads" => 1,
            )
        )
    end

    if is_cplex_available
        using CPLEX
        @time runtests(
            "CPLEX",
            optimizer_with_attributes(
                CPLEX.Optimizer,
                "CPXPARAM_Threads" => 1,
            ),
        )
    end
end
