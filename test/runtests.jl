#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Revise
using Test
using MIPLearn

includet("Cuts/BlackBox/test_cplex.jl")

function runtests()
    test_cuts_blackbox_cplex()
end
