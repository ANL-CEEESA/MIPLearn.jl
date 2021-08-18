#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Cbc
using JuMP
using MathOptInterface
using MIPLearn
const MOI = MathOptInterface

function find_lazy(model::Model, cb_data)::Vector{String}
    x = variable_by_name(model, "x")
    y = variable_by_name(model, "y")
    x_val = value(cb_data, x)
    y_val = value(cb_data, y)
    if x_val + y_val > 1 + 1e-6
        return ["con"]
    end
    return []
end

function enforce_lazy(model::Model, cb_data, violation::String)::Nothing
    if violation == "con"
        x = variable_by_name(model, "x")
        y = variable_by_name(model, "y")
        con = @build_constraint(x + y <= 1)
        submit(cb_data, con)
    end
    return
end

function build_model(data)
    model = Model()
    @variable(model, x, Bin)
    @variable(model, y, Bin)
    @objective(model, Max, 2 * x + y)
    @constraint(model, c1, x + y <= 2)
    @lazycb(model, find_lazy, enforce_lazy)
    return model
end

@testset "Lazy callback" begin
    @testset "JuMPInstance" begin
        model = build_model(nothing)
        instance = JuMPInstance(model)
        solver = LearningSolver(Cbc.Optimizer)
        solve!(solver, instance)
        @test value(model[:x]) == 1.0
        @test value(model[:y]) == 0.0
    end

    @testset "FileInstance" begin
        data = nothing
        filename = tempname()
        MIPLearn.save_data(filename, data)
        instance = FileInstance(filename, build_model)
        solver = LearningSolver(Cbc.Optimizer)
        solve!(solver, instance)
        h5 = MIPLearn.Hdf5Sample(filename)
        @test h5.get_array("mip_var_values") == [1.0, 0.0]
    end
end
