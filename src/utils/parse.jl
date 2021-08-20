#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

function parse_name(name::AbstractString)::Vector{String}
    # x
    m = match(r"^[-a-z0-9_]*$", name)
    if m !== nothing
        return [name]
    end

    # x[1,2,3]
    m = match(r"^([-a-z0-9_]*)\[([-a-z0-9_,]*)\]$"i, name)
    if m !== nothing
        return [m[1], split(m[2], ",")...]
    end

    # unknown
    error("Could not parse: $(name)")
end
