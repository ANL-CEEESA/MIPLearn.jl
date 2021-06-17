#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

module MIPLearn

using PyCall

global DynamicLazyConstraintsComponent = PyNULL()
global JuMPSolver = PyNULL()
global MinPrecisionThreshold = PyNULL()
global miplearn = PyNULL()
global ObjectiveValueComponent = PyNULL()
global PrimalSolutionComponent = PyNULL()
global PyFileInstance = PyNULL()
global PyJuMPInstance = PyNULL()
global StaticLazyConstraintsComponent = PyNULL()
global traceback = PyNULL()
global UserCutsComponent = PyNULL()

include("utils/log.jl")
include("utils/exceptions.jl")
include("instance/abstract.jl")
include("instance/jump.jl")
include("instance/file.jl")
include("solvers/jump.jl")
include("solvers/learning.jl")
include("solvers/macros.jl")
include("utils/benchmark.jl")

function __init__()
    copy!(miplearn, pyimport("miplearn"))
    copy!(traceback, pyimport("traceback"))
    copy!(DynamicLazyConstraintsComponent, miplearn.DynamicLazyConstraintsComponent)
    copy!(UserCutsComponent, miplearn.UserCutsComponent)
    copy!(ObjectiveValueComponent, miplearn.ObjectiveValueComponent)
    copy!(PrimalSolutionComponent, miplearn.PrimalSolutionComponent)
    copy!(StaticLazyConstraintsComponent, miplearn.StaticLazyConstraintsComponent)
    copy!(MinPrecisionThreshold, miplearn.MinPrecisionThreshold)
    __init_PyFileInstance__()
    __init_PyJuMPInstance__()
    __init_JuMPSolver__()
end

export DynamicLazyConstraintsComponent,
       UserCutsComponent,
       ObjectiveValueComponent,
       PrimalSolutionComponent,
       StaticLazyConstraintsComponent,
       MinPrecisionThreshold

end # module
