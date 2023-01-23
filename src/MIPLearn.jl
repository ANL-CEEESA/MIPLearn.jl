#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

module MIPLearn

using PyCall
using SparseArrays

global miplearn = PyNULL()
global Hdf5Sample = PyNULL()

to_str_array(values) = py"to_str_array"(values)

from_str_array(values) = py"from_str_array"(values)

function __init__()
    copy!(miplearn, pyimport("miplearn"))
    copy!(Hdf5Sample, miplearn.features.sample.Hdf5Sample)

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

include("Cuts/BlackBox/cplex.jl")

export Hdf5Sample

end # module