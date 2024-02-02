#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

module MIPLearnT

using Test
using Logging
using JuliaFormatter
using HiGHS
using Glob

BASEDIR = dirname(@__FILE__)
FIXTURES = "$BASEDIR/../fixtures"

include("fixtures.jl")

include("BB/test_bb.jl")
include("components/test_cuts.jl")
include("components/test_lazy.jl")
include("Cuts/BlackBox/test_cplex.jl")
include("problems/test_setcover.jl")
include("problems/test_stab.jl")
include("problems/test_tsp.jl")
include("solvers/test_jump.jl")
include("test_io.jl")
include("test_usage.jl")

function runtests()
    @testset "MIPLearn" begin
        @testset "BB" begin
            test_bb()
        end
        test_io()
        test_problems_setcover()
        test_problems_stab()
        test_problems_tsp()
        test_solvers_jump()
        test_usage()
        test_cuts()
        test_lazy()
    end
end

function format()
    JuliaFormatter.format(BASEDIR, verbose=true)
    JuliaFormatter.format("$BASEDIR/../../src", verbose=true)
    return
end


export runtests, format

end # module MIPLearnT
