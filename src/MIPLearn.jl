#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

module MIPLearn

using PyCall
using Requires

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
global MemorySample = PyNULL()
global Hdf5Sample = PyNULL()

include("solvers/structs.jl")

include("utils/log.jl")
include("utils/exceptions.jl")
include("instance/abstract_instance.jl")
include("instance/jump_instance.jl")
include("instance/file_instance.jl")
include("solvers/jump_solver.jl")
include("solvers/learning_solver.jl")
include("solvers/macros.jl")
include("utils/benchmark.jl")
include("utils/parse.jl")

include("bb/BB.jl")
include("cuts/Cuts.jl")

function __init__()
    copy!(miplearn, pyimport("miplearn"))
    copy!(traceback, pyimport("traceback"))
    copy!(DynamicLazyConstraintsComponent, miplearn.DynamicLazyConstraintsComponent)
    copy!(UserCutsComponent, miplearn.UserCutsComponent)
    copy!(ObjectiveValueComponent, miplearn.ObjectiveValueComponent)
    copy!(PrimalSolutionComponent, miplearn.PrimalSolutionComponent)
    copy!(StaticLazyConstraintsComponent, miplearn.StaticLazyConstraintsComponent)
    copy!(MinPrecisionThreshold, miplearn.MinPrecisionThreshold)
    copy!(MemorySample, miplearn.features.sample.MemorySample)
    copy!(Hdf5Sample, miplearn.features.sample.Hdf5Sample)
    __init_PyFileInstance__()
    __init_PyJuMPInstance__()
    __init_JuMPSolver__()

    py"""
    import numpy as np

    def to_str_array(values):
        if values is None:
            return None
        return np.array(values, dtype="S")

    def from_str_array(values):
        return [v.decode() for v in values]
    """
end

to_str_array(values) = py"to_str_array"(values)
from_str_array(values) = py"from_str_array"(values)

function convert(::Type{SparseMatrixCSC}, o::PyObject)
    I, J, V = pyimport("scipy.sparse").find(o)
    return sparse(I .+ 1, J .+ 1, V, o.shape...)
end

function PyObject(m::SparseMatrixCSC)
    pyimport("scipy.sparse").csc_matrix(
        (m.nzval, m.rowval .- 1, m.colptr .- 1),
        shape = size(m),
    ).tocoo()
end

export DynamicLazyConstraintsComponent,
    UserCutsComponent,
    ObjectiveValueComponent,
    PrimalSolutionComponent,
    StaticLazyConstraintsComponent,
    MinPrecisionThreshold,
    Hdf5Sample

end # module
