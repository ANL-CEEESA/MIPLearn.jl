#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Printf
using JLD2

global H5File = PyNULL()
global write_pkl_gz = PyNULL()
global read_pkl_gz = PyNULL()

to_str_array(values) = py"to_str_array"(values)

from_str_array(values) = py"from_str_array"(values)

function __init_io__()
    copy!(H5File, pyimport("miplearn.h5").H5File)
    copy!(write_pkl_gz, pyimport("miplearn.io").write_pkl_gz)
    copy!(read_pkl_gz, pyimport("miplearn.io").read_pkl_gz)

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
        shape=size(m),
    ).tocoo()
end

function write_jld2(
    objs::Vector,
    dirname::AbstractString;
    prefix::AbstractString=""
)::Vector{String}
    mkpath(dirname)
    filenames = [@sprintf("%s/%s%05d.jld2", dirname, prefix, i) for i = 1:length(objs)]
    for (i, obj) in enumerate(objs)
        jldsave(filenames[i]; obj)
    end
    return filenames
end

function read_jld2(filename::AbstractString)::Any
    jldopen(filename, "r") do file
        return file["obj"]
    end
end

export H5File, write_pkl_gz, read_pkl_gz, write_jld2, read_jld2
