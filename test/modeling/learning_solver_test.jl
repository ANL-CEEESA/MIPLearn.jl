#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using MIPLearn
using Gurobi

@testset "macros" begin
    weights = [1.0, 2.0, 3.0]
    prices = [5.0, 6.0, 7.0]
    capacity = 3.0

    # Create standard JuMP model
    model = Model()
    n = length(weights)
    @variable(model, x[1:n], Bin)
    @objective(model, Max, sum(x[i] * prices[i] for i in 1:n))
    @constraint(model, c1, sum(x[i] * weights[i] for i in 1:n) <= capacity)

    # Add ML information to the model
    @feature(model, [5.0])
    @feature(c1, [1.0, 2.0, 3.0])
    @category(c1, "c1")
    for i in 1:n
        @feature(x[i], [weights[i]; prices[i]])
        @category(x[i], "type-$i")
    end
 
    # Should store ML information
    @test model.ext[:miplearn][:variable_features][x[1]] == [1.0, 5.0]
    @test model.ext[:miplearn][:variable_features][x[2]] == [2.0, 6.0]
    @test model.ext[:miplearn][:variable_features][x[3]] == [3.0, 7.0]
    @test model.ext[:miplearn][:variable_categories][x[1]] == "type-1"
    @test model.ext[:miplearn][:variable_categories][x[2]] == "type-2"
    @test model.ext[:miplearn][:variable_categories][x[3]] == "type-3"
    @test model.ext[:miplearn][:constraint_features][c1] == [1.0, 2.0, 3.0]
    @test model.ext[:miplearn][:constraint_categories][c1] == "c1"
    @test model.ext[:miplearn][:instance_features] == [5.0]

    solver = LearningSolver(Gurobi.Optimizer)
    instance = JuMPInstance(model)
    stats = solve!(solver, instance)
    @test stats["mip_lower_bound"] == 11.0
    @test length(instance.py.samples) == 1
    fit!(solver, [instance])
    solve!(solver, instance)
end
