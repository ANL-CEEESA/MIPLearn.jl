module MIPLearnT

using Test
using Logging
using JuliaFormatter
using HiGHS

BASEDIR = dirname(@__FILE__)
FIXTURES = "$BASEDIR/../fixtures"

include("Cuts/BlackBox/test_cplex.jl")
include("problems/test_setcover.jl")
include("test_h5.jl")
include("solvers/test_jump.jl")

function runtests()
    @testset "MIPLearn" begin
        test_cuts_blackbox_cplex()
        test_h5()
        test_problems_setcover()
        test_solvers_jump()
    end
end

function format()
    JuliaFormatter.format(BASEDIR, verbose = true)
    JuliaFormatter.format("$BASEDIR/../../src", verbose = true)
    return
end


export runtests, format

end # module MIPLearnT
