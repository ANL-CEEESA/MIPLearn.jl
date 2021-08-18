#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Test
using MIPLearn

MIPLearn.setup_logger()

@testset "MIPLearn" begin
    include("fixtures/knapsack.jl")
    include("instance/file_instance_test.jl")
    include("instance/jump_instance_test.jl")
    include("solvers/jump_solver_test.jl")
    include("solvers/learning_solver_test.jl")
    include("utils/benchmark_test.jl")
end
