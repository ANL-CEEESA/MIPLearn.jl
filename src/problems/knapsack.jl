#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP

function knapsack_model(
    weights::Array{Float64, 1},
    prices::Array{Float64, 1},
    capacity::Float64,
)
    model = Model()
    n = length(weights)
    @variable(model, x[0:(n-1)], Bin)
    @objective(model, Max, sum(x[i] * prices[i+1] for i in 0:(n-1)))
    @constraint(
        model,
        eq_capacity,
        sum(
            x[i] * weights[i+1]
            for i in 0:(n-1)
        ) <= capacity,
    )
    return model
end
