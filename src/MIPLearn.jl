#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

module MIPLearn

using PyCall
using SparseArrays

include("collectors.jl")
include("components.jl")
include("extractors.jl")
include("io.jl")
include("problems/setcover.jl")
include("solvers/jump.jl")
include("solvers/learning.jl")

function __init__()
    __init_collectors__()
    __init_components__()
    __init_extractors__()
    __init_io__()
    __init_problems_setcover__()
    __init_solvers_jump__()
    __init_solvers_learning__()
end

include("BB/BB.jl")
include("Cuts/BlackBox/cplex.jl")

end # module
