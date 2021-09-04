#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP
import JSON

Base.@kwdef mutable struct JuMPInstance <: Instance
    py::Union{Nothing,PyCall.PyObject} = nothing
    model::Union{Nothing,JuMP.Model} = nothing
    samples::Vector{PyCall.PyObject} = []

    function JuMPInstance(model::JuMP.Model)::JuMPInstance
        init_miplearn_ext(model)
        instance = new(nothing, model, [])
        py = PyJuMPInstance(instance)
        instance.py = py
        return instance
    end
end

function to_model(instance::JuMPInstance)::JuMP.Model
    return instance.model
end

function get_instance_features(instance::JuMPInstance)::Union{Vector{Float64},Nothing}
    return instance.model.ext[:miplearn]["instance_features"]
end

function _concat_features(dict, names)::Matrix{Float64}
    if isempty(dict)
        return zeros(length(names), 1)
    end
    ncols = length(first(dict).second)
    return vcat([n in keys(dict) ? dict[n]' : zeros(ncols) for n in names]...)
end

function _concat_categories(dict, names)::Vector{String}
    return String[n in keys(dict) ? dict[n] : n for n in names]
end

function get_variable_features(
    instance::JuMPInstance,
    names::Vector{String},
)::Matrix{Float64}
    return _concat_features(instance.model.ext[:miplearn]["variable_features"], names)
end

function get_variable_categories(instance::JuMPInstance, names::Vector{String})
    return _concat_categories(instance.model.ext[:miplearn]["variable_categories"], names)
end

function get_constraint_features(
    instance::JuMPInstance,
    names::Vector{String},
)::Matrix{Float64}
    return _concat_features(instance.model.ext[:miplearn]["constraint_features"], names)
end

function get_constraint_categories(instance::JuMPInstance, names::Vector{String})
    return _concat_categories(instance.model.ext[:miplearn]["constraint_categories"], names)
end

get_samples(instance::JuMPInstance) = instance.samples

function create_sample!(instance::JuMPInstance)
    sample = MemorySample()
    push!(instance.samples, sample)
    return sample
end

function find_violated_lazy_constraints(instance::JuMPInstance, solver)::Vector{String}
    if "lazy_find_cb" ∈ keys(instance.model.ext[:miplearn])
        return instance.model.ext[:miplearn]["lazy_find_cb"](instance.model, solver.data)
    else
        return []
    end
end

function enforce_lazy_constraint(instance::JuMPInstance, solver, violation::String)::Nothing
    instance.model.ext[:miplearn]["lazy_enforce_cb"](instance.model, solver.data, violation)
end

function solve!(solver::LearningSolver, model::JuMP.Model; kwargs...)
    solve!(solver, JuMPInstance(model); kwargs...)
end

function __init_PyJuMPInstance__()
    @pydef mutable struct Class <: miplearn.Instance
        function __init__(self, jl)
            self.jl = jl
        end
        to_model(self) = to_model(self.jl)
        get_instance_features(self) = get_instance_features(self.jl)
        get_variable_features(self, names) =
            get_variable_features(self.jl, from_str_array(names))
        get_variable_categories(self, names) =
            to_str_array(get_variable_categories(self.jl, from_str_array(names)))
        get_constraint_features(self, names) =
            get_constraint_features(self.jl, from_str_array(names))
        get_constraint_categories(self, names) =
            to_str_array(get_constraint_categories(self.jl, from_str_array(names)))
        get_samples(self) = get_samples(self.jl)
        create_sample(self) = create_sample!(self.jl)
        find_violated_lazy_constraints(self, solver, _) =
            find_violated_lazy_constraints(self.jl, solver)
        enforce_lazy_constraint(self, solver, _, violation) =
            enforce_lazy_constraint(self.jl, solver, violation)
    end
    copy!(PyJuMPInstance, Class)
end

export JuMPInstance, save, load_instance
