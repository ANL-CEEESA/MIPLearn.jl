#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using PackageCompiler

using Cbc
using Clp
using Conda
using CSV
using DataFrames
using Distributed
using JLD2
using JSON
using JuMP
using Logging
using MathOptInterface
using Printf
using PyCall
using TimerOutputs

pkg = [
    :Cbc
    :Clp
    :Conda
    :CSV
    :DataFrames
    :Distributed
    :JLD2
    :JSON
    :JuMP
    :Logging
    :MathOptInterface
    :Printf
    :PyCall
    :TimerOutputs
]

@info "Building system image..."
create_sysimage(
    pkg,
    precompile_statements_file = "build/precompile.jl",
    sysimage_path = "build/sysimage.so",
)
