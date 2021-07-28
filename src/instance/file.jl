#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import Base: flush

mutable struct FileInstance <: Instance
    py::Union{Nothing,PyCall.PyObject}
    loaded::Union{Nothing, JuMPInstance}
    filename::AbstractString

    function FileInstance(filename::String)::FileInstance
        instance = new(nothing, nothing, filename)
        instance.py = PyFileInstance(instance)
        return instance
    end
end

to_model(instance::FileInstance) = to_model(instance.loaded)
get_instance_features(instance::FileInstance) = get_instance_features(instance.loaded)
get_variable_features(instance::FileInstance) = get_variable_features(instance.loaded)
get_variable_categories(instance::FileInstance) = get_variable_categories(instance.loaded)
get_constraint_features(instance::FileInstance) = get_constraint_features(instance.loaded)
get_samples(instance::FileInstance) = get_samples(instance.loaded)
push_sample!(instance::FileInstance, sample::PyCall.PyObject) = push_sample!(instance.loaded, sample)

function get_constraint_categories(instance::FileInstance)
    return get_constraint_categories(instance.loaded)
end

function load(instance::FileInstance)
    if instance.loaded === nothing
        instance.loaded = load_instance(instance.filename)
    end
end

function free(instance::FileInstance)
    instance.loaded.samples = []
    instance.loaded = nothing
    GC.gc()
end

function flush(instance::FileInstance)
    save(instance.filename, instance.loaded)
end

function __init_PyFileInstance__()
    @pydef mutable struct Class <: miplearn.Instance
        function __init__(self, jl)
            self.jl = jl
        end
        to_model(self) = to_model(self.jl)
        get_instance_features(self) = get_instance_features(self.jl)
        get_variable_features(self) = get_variable_features(self.jl)
        get_variable_categories(self) = get_variable_categories(self.jl)
        get_constraint_features(self) = get_constraint_features(self.jl)
        get_constraint_categories(self) = get_constraint_categories(self.jl)
        get_samples(self) = get_samples(self.jl)
        push_sample(self, sample) = push_sample!(self.jl, sample)
        load(self) = load(self.jl)
        free(self) = free(self.jl)
        flush(self) = flush(self.jl)
    end
    copy!(PyFileInstance, Class)
end

export FileInstance
