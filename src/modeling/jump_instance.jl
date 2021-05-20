#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP


@pydef mutable struct PyJuMPInstance <: miplearn.Instance
    function __init__(self, model)
        self.model = model
        self.samples = []
    end

    function to_model(self)
        return self.model
    end

    function get_instance_features(self)
        return self.model.ext[:miplearn][:instance_features]
    end

    function get_variable_features(self, var_name)
        model = self.model
        v = variable_by_name(model, var_name)
        return model.ext[:miplearn][:variable_features][v]
    end

    function get_variable_category(self, var_name)
        model = self.model
        v = variable_by_name(model, var_name)
        return model.ext[:miplearn][:variable_categories][v]
    end

    function get_constraint_features(self, cname)
        model = self.model
        c = constraint_by_name(model, cname)
        return model.ext[:miplearn][:constraint_features][c]
    end

    function get_constraint_category(self, cname)
        model = self.model
        c = constraint_by_name(model, cname)
        return model.ext[:miplearn][:constraint_categories][c]
    end
end


struct JuMPInstance
    py::PyCall.PyObject
end


function JuMPInstance(model::Model)
    return JuMPInstance(PyJuMPInstance(model))
end


export JuMPInstance
