#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

struct FileInstance
    filename::AbstractString
    loaded::Union{Nothing,JuMPInstance}
end


function FileInstance(filename::AbstractString)::FileInstance
    return FileInstance(
        filename,
        nothing,
    )
end


function load!(instance::FileInstance)
    instance.loaded = load_jump_instance(instance.filename)
end


function free!(instance::FileInstance)
    instance.loaded = nothing
end


function flush!(instance::FileInstance)
    save(instance.filename, instance.loaded)
end
