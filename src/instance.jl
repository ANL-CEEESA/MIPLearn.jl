#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JSON2
import Base: dump

to_model(instance) =
    error("not implemented: to_model")

get_instance_features(instance) = [0.0]

get_variable_features(instance, varname) = [0.0]

get_variable_category(instance, varname) = "default"

find_violated_lazy_constraints(instance, model) = []

build_lazy_constraint(instance, model, v) = nothing

macro Instance(klass)
    quote
        @pydef mutable struct Wrapper <: Instance
            function __init__(self, args...; kwargs...)
                self.data = $(esc(klass))(args...; kwargs...)
                self.training_data = []
                self.features = miplearn.Features()
            end
                
            to_model(self) =
                $(esc(:to_model))(self.data)
                
            get_instance_features(self) =
                $(esc(:get_instance_features))(self.data)
                
            get_variable_features(self, varname) =
                $(esc(:get_variable_features))(self.data, varname)

            get_variable_category(self, varname) =
                $(esc(:get_variable_category))(self.data, varname)

            find_violated_lazy_constraints(self, model) =
                find_violated_lazy_constraints(self.data, model)
            
            build_lazy_constraint(self, model, v) =
                build_lazy_constraint(self.data, model, v)
            
            load(self) = nothing

            flush(self) = nothing
        end
    end
end

export get_instance_features,
       get_variable_features,
       get_variable_category,
       find_violated_lazy_constraints,
       build_lazy_constraint,
       to_model,
       dump,
       load!,
       @Instance