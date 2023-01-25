#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using HDF5
using MIPLearn

function test_cuts_blackbox_cplex()
    # Prepare filenames
    mps_filename = joinpath(@__DIR__, "../../fixtures/bell5.mps.gz")
    h5_filename = replace(mps_filename, ".mps.gz" => ".h5")

    # Run collector
    MIPLearn.collect(mps_filename, CplexBlackBoxCuts())

    # Read HDF5 file
    h5 = Hdf5Sample(h5_filename)
    rhs = h5.get_array("cuts_cpx_rhs")
    h5.file.close()
    @test length(rhs) > 0
end
