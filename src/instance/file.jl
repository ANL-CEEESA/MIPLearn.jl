#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.


@pydef mutable struct PyFileInstance <: miplearn.Instance
    function __init__(self, filename)
        self.filename = filename
        self.loaded = nothing
        self.samples = nothing
    end

    function to_model(self)
        return self.loaded.py.to_model()
    end

    function get_instance_features(self)
        return self.loaded.py.get_instance_features()
    end

    function get_variable_features(self, var_name)
        return self.loaded.py.get_variable_features(var_name)
    end

    function get_variable_category(self, var_name)
        return self.loaded.py.get_variable_category(var_name)
    end

    function get_constraint_features(self, cname)
        return self.loaded.py.get_constraint_features(cname)
    end

    function get_constraint_category(self, cname)
        return self.loaded.py.get_constraint_category(cname)
    end

    function load(self)
        if self.loaded === nothing
            self.loaded = load_instance(self.filename)
            self.samples = self.loaded.py.samples
        end
    end

    function free(self)
        self.loaded = nothing
        self.samples = nothing
    end

    function flush(self)
        self.loaded.py.samples = self.samples
        save(self.filename, self.loaded)
    end
end


struct FileInstance <: Instance
    py::PyCall.PyObject
    filename::AbstractString
end


function FileInstance(filename)::FileInstance
    filename isa AbstractString || error("filename should be a string. Found $(typeof(filename)) instead.")
    return FileInstance(
        PyFileInstance(filename),
        filename,
    )
end


export FileInstance
