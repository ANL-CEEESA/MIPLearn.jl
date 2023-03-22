#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

global BasicCollector = PyNULL()

function __init_collectors__()
    copy!(BasicCollector, pyimport("miplearn.collectors.basic").BasicCollector)
end

export BasicCollector
