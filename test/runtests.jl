#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Test
using Requires
using MIPLearn
MIPLearn.setup_logger()

is_cplex_available = false
@require CPLEX = "a076750e-1247-5638-91d2-ce28b192dca0" begin
    is_cplex_available = true
end

is_gurobi_available = false
@require Gurobi = "2e9cd046-0924-5485-92f1-d5272153d98b" begin
    is_gurobi_available = true
end

@testset "MIPLearn" begin
    include("fixtures/knapsack.jl")
    include("instance/file_instance_test.jl")
    include("instance/jump_instance_test.jl")
    include("solvers/jump_solver_test.jl")
    include("solvers/learning_solver_test.jl")
    include("utils/parse_test.jl")
    include("bb/lp_test.jl")
end
