#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using HDF5

function test_cuts_blackbox_cplex()
    # Prepare filenames
    mps_filename = joinpath(@__DIR__, "../../fixtures/bell5.mps.gz")
    h5_filename = replace(mps_filename, ".mps.gz" => ".h5")

    # Run collector
    MIPLearn.collect(mps_filename, CplexBlackBoxCuts())

    # Read HDF5 file
    h5open(h5_filename, "r+") do h5
        @test size(h5["cuts_cpx_lhs"]) == (12, 104)
        @test size(h5["cuts_cpx_rhs"]) == (12,)
    end
end
