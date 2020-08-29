using PackageCompiler

using JSON2
using CPLEX
using Gurobi
using JuMP
using MathOptInterface
using PyCall
using TimerOutputs

pkg = [:JSON2
       :CPLEX
       :Gurobi
       :JuMP
       :MathOptInterface
       :PyCall
       :TimerOutputs]

create_sysimage(pkg, sysimage_path="build/sysimage.so")