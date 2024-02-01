#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

function fixture_setcover_data()
    return SetCoverData(
        costs = [5, 10, 12, 6, 8],
        incidence_matrix = [
            1 0 0 1 0
            1 1 0 0 0
            0 0 1 1 1
        ],
    )
end

function fixture_setcover_model()
    return build_setcover_model_jump(fixture_setcover_data())
end
