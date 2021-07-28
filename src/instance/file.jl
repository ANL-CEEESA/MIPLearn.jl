#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

import Base: flush

mutable struct FileInstance <: Instance
    py::Union{Nothing,PyCall.PyObject}
    loaded::Union{Nothing, JuMPInstance}
    filename::AbstractString
    h5::PyCall.PyObject

    function FileInstance(filename::AbstractString)::FileInstance
        instance = new(nothing, nothing, filename)
        instance.py = PyFileInstance(instance)
        instance.h5 = Hdf5Sample(filename)
        instance.filename = filename
        return instance
    end
end

to_model(instance::FileInstance) = to_model(instance.loaded)
get_instance_features(instance::FileInstance) = get_instance_features(instance.loaded)
get_variable_features(instance::FileInstance) = get_variable_features(instance.loaded)
get_variable_categories(instance::FileInstance) = get_variable_categories(instance.loaded)
get_constraint_features(instance::FileInstance) = get_constraint_features(instance.loaded)

function get_samples(instance::FileInstance)
    return [instance.h5]
end

function create_sample!(instance::FileInstance)
    return instance.h5
end

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
        create_sample(self) = create_sample!(self.jl)
        load(self) = load(self.jl)
        free(self) = free(self.jl)
        flush(self) = flush(self.jl)
    end
    copy!(PyFileInstance, Class)
end

export FileInstance
