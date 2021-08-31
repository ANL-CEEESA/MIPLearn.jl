#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JLD2
import Base: flush

mutable struct FileInstance <: Instance
    py::Union{Nothing,PyCall.PyObject}
    loaded::Union{Nothing,JuMPInstance}
    filename::AbstractString
    h5::PyCall.PyObject
    build_model::Function

    function FileInstance(filename::AbstractString, build_model::Function)::FileInstance
        instance = new(nothing, nothing, filename, nothing, build_model)
        instance.py = PyFileInstance(instance)
        instance.h5 = Hdf5Sample("$filename.h5", mode="a")
        instance.filename = filename
        return instance
    end
end

to_model(instance::FileInstance) = to_model(instance.loaded)

get_instance_features(instance::FileInstance) = get_instance_features(instance.loaded)

get_variable_features(instance::FileInstance, names) =
    get_variable_features(instance.loaded, names)

get_variable_categories(instance::FileInstance, names) =
    get_variable_categories(instance.loaded, names)

get_constraint_features(instance::FileInstance, names) =
    get_constraint_features(instance.loaded, names)

get_constraint_categories(instance::FileInstance, names) =
    get_constraint_categories(instance.loaded, names)

find_violated_lazy_constraints(instance::FileInstance, solver) =
    find_violated_lazy_constraints(instance.loaded, solver)

enforce_lazy_constraint(instance::FileInstance, solver, violation) =
    enforce_lazy_constraint(instance.loaded, solver, violation)

function get_samples(instance::FileInstance)
    return [instance.h5]
end

function create_sample!(instance::FileInstance)
    return instance.h5
end

function load(instance::FileInstance)
    if instance.loaded === nothing
        data = load_data(instance.filename)
        instance.loaded = JuMPInstance(instance.build_model(data))
    end
end

function free(instance::FileInstance)
    instance.loaded.samples = []
    instance.loaded = nothing
    GC.gc()
end

function save_data(filename::AbstractString, data)::Nothing
    jldsave(filename, data = data)
end

function load_data(filename::AbstractString)
    jldopen(filename, "r") do file
        return file["data"]
    end
end

function flush(instance::FileInstance) end

function __init_PyFileInstance__()
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
        load(self) = load(self.jl)
        free(self) = free(self.jl)
        flush(self) = flush(self.jl)
        find_violated_lazy_constraints(self, solver, _) =
            find_violated_lazy_constraints(self.jl, solver)
        enforce_lazy_constraint(self, solver, _, violation) =
            enforce_lazy_constraint(self.jl, solver, violation)
    end
    copy!(PyFileInstance, Class)
end

export FileInstance
