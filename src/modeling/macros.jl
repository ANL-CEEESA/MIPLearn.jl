#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

function init_miplearn_ext(model)::Dict
    if :miplearn ∉ keys(model.ext)
        model.ext[:miplearn] = Dict{Symbol,Any}(
            :features => Dict(
                :variables => Dict{String,Dict}(),
                :constraints => Dict{String,Dict}(),
                :instance => Dict{Symbol,Any}(),
            ),
            :training_samples => [],
        )
    end
    return model.ext[:miplearn]
end


function init_miplearn_ext(v::VariableRef)::Dict
    ext = init_miplearn_ext(v.model)
    if name(v) ∉ keys(ext[:features][:variables])
        ext[:features][:variables][name(v)] = Dict{Symbol,Any}()
    end
    return ext
end


function init_miplearn_ext(c::ConstraintRef)::Dict
    ext = init_miplearn_ext(c.model)
    if name(c) ∉ keys(ext[:features][:constraints])
        ext[:features][:constraints][name(c)] = Dict{Symbol,Any}()
    end
    return ext
end


function set_features!(m::Model, f::Array{Float64})::Nothing
    ext = init_miplearn_ext(m)
    ext[:features][:instance][:user_features] = f
    return
end


function set_features!(v::VariableRef, f::Array{Float64})::Nothing
    ext = init_miplearn_ext(v)
    ext[:features][:variables][name(v)][:user_features] = f
    return
end


function set_category!(v::VariableRef, category::String)::Nothing
    ext = init_miplearn_ext(v)
    ext[:features][:variables][name(v)][:category] = category
    return
end


function set_features!(c::ConstraintRef, f::Array{Float64})::Nothing
    ext = init_miplearn_ext(c)
    ext[:features][:constraints][name(c)][:user_features] = f
    return
end


function set_category!(c::ConstraintRef, category::String)::Nothing
    ext = init_miplearn_ext(c)
    ext[:features][:constraints][name(c)][:category] = category
    return
end


macro feature(obj, features)
    quote
        set_features!($(esc(obj)), $(esc(features)))
    end
end


macro category(obj, category)
    quote
        set_category!($(esc(obj)), $(esc(category)))
    end
end
