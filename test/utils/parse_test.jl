#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using MIPLearn

@testset "Parse" begin
    @test MIPLearn.parse_name("x") == ["x"]
    @test MIPLearn.parse_name("x[3]") == ["x", "3"]
    @test MIPLearn.parse_name("test_eq[x]") == ["test_eq", "x"]
    @test MIPLearn.parse_name("test_eq[x,y,z]") == ["test_eq", "x", "y", "z"]
end
