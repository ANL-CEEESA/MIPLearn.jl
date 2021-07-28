#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

function init_miplearn_ext(model)::Dict
    if :miplearn ∉ keys(model.ext)
        model.ext[:miplearn] = Dict()
        model.ext[:miplearn]["instance_features"] = [0.0]
        model.ext[:miplearn]["variable_features"] = Dict{AbstractString, Vector{Float64}}()
        model.ext[:miplearn]["variable_categories"] = Dict{AbstractString, String}()
        model.ext[:miplearn]["constraint_features"] = Dict{AbstractString, Vector{Float64}}()
        model.ext[:miplearn]["constraint_categories"] = Dict{AbstractString, String}()
    end
    return model.ext[:miplearn]
end


function set_features!(m::Model, f::Array{Float64})::Nothing
    ext = init_miplearn_ext(m)
    ext["instance_features"] = f
    return
end


function set_features!(v::VariableRef, f::Array{Float64})::Nothing
    ext = init_miplearn_ext(v.model)
    n = _get_and_check_name(v)
    ext["variable_features"][n] = f
    return
end


function set_category!(v::VariableRef, category::String)::Nothing
    ext = init_miplearn_ext(v.model)
    n = _get_and_check_name(v)
    ext["variable_categories"][n] = category
    return
end


function set_features!(c::ConstraintRef, f::Array{Float64})::Nothing
    ext = init_miplearn_ext(c.model)
    n = _get_and_check_name(c)
    ext["constraint_features"][n] = f
    return
end


function set_category!(c::ConstraintRef, category::String)::Nothing
    ext = init_miplearn_ext(c.model)
    n = _get_and_check_name(c)
    ext["constraint_categories"][n] = category
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

function _get_and_check_name(obj)
    n = name(obj)
    length(n) > 0 || error(
        "Features and categories can only be assigned to variables and " *
        "constraints that have names. Unnamed model element detected."
    )
    return n
end


export @feature,
       @category
