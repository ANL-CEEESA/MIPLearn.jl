#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using MIPLearn
using Cbc

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

    # Add machine-learning information
    @feature(model, [5.0])
    @feature(c1, [1.0, 2.0, 3.0])
    @category(c1, "c1")
    for i in 1:n
        @feature(x[i], [weights[i]; prices[i]])
        @category(x[i], "type-$i")
    end
 
    # Should store variable features
    @test model.ext[:miplearn][:features][:variables] == Dict(
        "x[1]" => Dict(
            :user_features => [1.0, 5.0],
            :category => "type-1",

        ),
        "x[2]" => Dict(
            :user_features => [2.0, 6.0],
            :category => "type-2",
        ),
        "x[3]" => Dict(
            :user_features => [3.0, 7.0],
            :category => "type-3",
        ),
    )

    # Should store constraint features
    @test model.ext[:miplearn][:features][:constraints] == Dict(
        "c1" => Dict(
            :user_features => [1.0, 2.0, 3.0],
            :category => "c1",
        )
    )

    # Should store instance features
    @test model.ext[:miplearn][:features][:instance] == Dict(
        :user_features => [5.0],
    )

    solver = LearningSolver(optimizer=Cbc.Optimizer)
    
    # Should return correct stats
    stats = solve!(solver, model)
    @test stats["Lower bound"] == 11.0

    # Should add a sample to the training data
    @test length(model.ext[:miplearn][:training_samples]) == 1
    sample = model.ext[:miplearn][:training_samples][1]
    @test sample["lower_bound"] == 11.0
    @test sample["solution"]["x[1]"] == 1.0

    fit!(solver, [model])

    solve!(solver, model)
end
