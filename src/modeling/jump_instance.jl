#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

@pydef mutable struct JuMPInstance <: miplearn.Instance
    function __init__(self, model)
        self.model = model

        # init_miplearn_ext(model)
        # features = model.ext[:miplearn][:features]
        # # Copy training data
        # training_data = []
        # for sample in self.model.ext[:miplearn][:training_samples]
        #     pysample = miplearn.TrainingSample()
        #     pysample.__dict__ = sample
        #     push!(training_data, pysample)
        # end
        # self.training_data = training_data

        # # Copy features to data classes
        # self.features = miplearn.Features(
        #     instance=miplearn.InstanceFeatures(
        #         user_features=PyCall.array2py(
        #             features[:instance][:user_features],
        #         ),
        #         lazy_constraint_count=0,
        #     ),
        #     variables=Dict(
        #         varname => miplearn.VariableFeatures(
        #             category=vfeatures[:category],
        #             user_features=PyCall.array2py(
        #                 vfeatures[:user_features],
        #             ),
        #         )
        #         for (varname, vfeatures) in features[:variables]
        #     ),
        #     constraints=Dict(
        #         cname => miplearn.ConstraintFeatures(
        #             category=cfeat[:category],
        #             user_features=PyCall.array2py(
        #                 cfeat[:user_features],
        #             ),
        #         )
        #         for (cname, cfeat) in features[:constraints]
        #     ),
        # )
    end

    function to_model(self)
        return self.model
    end
end
