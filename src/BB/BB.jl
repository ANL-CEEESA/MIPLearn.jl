#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

module BB

using Requires

frac(x) = x - floor(x)

include("structs.jl")

include("collect.jl")
include("nodepool.jl")
include("optimize.jl")
include("log.jl")
include("lp.jl")
include("varbranch/hybrid.jl")
include("varbranch/infeasibility.jl")
include("varbranch/pseudocost.jl")
include("varbranch/random.jl")
include("varbranch/reliability.jl")
include("varbranch/strong.jl")

function __init__()
    @require CPLEX = "a076750e-1247-5638-91d2-ce28b192dca0" include("cplex.jl")
end

end # module
