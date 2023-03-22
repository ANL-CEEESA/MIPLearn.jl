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
include("Cuts/BlackBox/test_cplex.jl")
include("problems/test_setcover.jl")
include("test_io.jl")
include("solvers/test_jump.jl")
include("test_usage.jl")

function runtests()
    @testset "MIPLearn" begin
        @testset "BB" begin
            test_bb()
        end
        test_cuts_blackbox_cplex()
        test_io()
        test_problems_setcover()
        test_solvers_jump()
        test_usage()
    end
end

function format()
    JuliaFormatter.format(BASEDIR, verbose = true)
    JuliaFormatter.format("$BASEDIR/../../src", verbose = true)
    return
end


export runtests, format

end # module MIPLearnT
