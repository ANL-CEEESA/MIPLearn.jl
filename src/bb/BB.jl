#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

module BB

frac(x) = x - floor(x)

include("structs.jl")

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

end # module
