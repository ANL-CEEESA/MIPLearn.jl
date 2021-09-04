#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JLD2
import Base: flush

mutable struct FileInstance <: Instance
    py::Union{Nothing,PyCall.PyObject}
    loaded::Union{Nothing,JuMPInstance}
    filename::AbstractString
    sample::PyCall.PyObject
    build_model::Function
    mode::String

    function FileInstance(
        filename::AbstractString,
        build_model::Function;
        mode::String = "a",
    )::FileInstance
        instance = new(nothing, nothing, filename, nothing, build_model, mode)
        instance.py = PyFileInstance(instance)
        if mode != "r" || isfile("$filename.h5")
            instance.sample = Hdf5Sample("$filename.h5", mode = mode)
        end
        instance.filename = filename
        return instance
    end
end

function _load!(instance::FileInstance)
    if instance.loaded === nothing
        data = load_data(instance.filename)
        instance.loaded = JuMPInstance(instance.build_model(data))
    end
end

function free(instance::FileInstance)
    instance.loaded = nothing
end

function to_model(instance::FileInstance)
    _load!(instance)
    return to_model(instance.loaded)
end

function get_instance_features(instance::FileInstance)
    _load!(instance)
    return get_instance_features(instance.loaded)
end

function get_variable_features(instance::FileInstance, names)
    _load!(instance)
    return get_variable_features(instance.loaded, names)
end

function get_variable_categories(instance::FileInstance, names)
    _load!(instance)
    return get_variable_categories(instance.loaded, names)
end

function get_constraint_features(instance::FileInstance, names)
    _load!(instance)
    return get_constraint_features(instance.loaded, names)
end

function get_constraint_categories(instance::FileInstance, names)
    _load!(instance)
    return get_constraint_categories(instance.loaded, names)
end

function find_violated_lazy_constraints(instance::FileInstance, solver)
    _load!(instance)
    return find_violated_lazy_constraints(instance.loaded, solver)
end

function enforce_lazy_constraint(instance::FileInstance, solver, violation)
    _load!(instance)
    return enforce_lazy_constraint(instance.loaded, solver, violation)
end

function get_samples(instance::FileInstance)
    return [instance.sample]
end

function create_sample!(instance::FileInstance)
    if instance.mode == "r"
        return MemorySample()
    else
        return instance.sample
    end
end

function save_data(filename::AbstractString, data)::Nothing
    jldsave(filename, data = data)
end

function load_data(filename::AbstractString)
    jldopen(filename, "r") do file
        return file["data"]
    end
end

function load(filename::AbstractString, build_model::Function)
    jldopen(filename, "r") do file
        return build_model(file["data"])
    end
end

function save(data::AbstractVector, dirname::String)::Nothing
    mkpath(dirname)
    for (i, d) in enumerate(data)
        filename = joinpath(dirname, @sprintf("%06d.jld2", i))
        jldsave(filename, data = d)
    end
end

function solve!(
    solver::LearningSolver,
    filenames::Vector,
    build_model::Function;
    tee::Bool = false,
)
    for filename in filenames
        solve!(solver, filename, build_model; tee)
    end
end

function fit!(
    solver::LearningSolver,
    filenames::Vector,
    build_model::Function;
    tee::Bool = false,
)
    instances = [FileInstance(f, build_model) for f in filenames]
    fit!(solver, instances)
end

function solve!(
    solver::LearningSolver,
    filename::AbstractString,
    build_model::Function;
    tee::Bool = false,
)
    solve!(solver, FileInstance(filename, build_model); tee)
end

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
        get_samples(self) = get_samples(self.jl)
        create_sample(self) = create_sample!(self.jl)
        find_violated_lazy_constraints(self, solver, _) =
            find_violated_lazy_constraints(self.jl, solver)
        enforce_lazy_constraint(self, solver, _, violation) =
            enforce_lazy_constraint(self.jl, solver, violation)
        free(self) = free(self.jl)

        # FIXME: The two functions below are disabled because they break lazy loading
        # of FileInstance.

        # get_constraint_features(self, names) =
        #     get_constraint_features(self.jl, from_str_array(names))
        # get_constraint_categories(self, names) =
        #     to_str_array(get_constraint_categories(self.jl, from_str_array(names)))

    end
    copy!(PyFileInstance, Class)
end

export FileInstance
