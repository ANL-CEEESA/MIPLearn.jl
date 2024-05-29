#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2024, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using HiGHS

function test_cuts_tableau_gmi()
    mps_filename = "$BASEDIR/../fixtures/bell5.mps.gz"
    h5_filename = "$BASEDIR/../fixtures/bell5.h5"
    collect_gmi(mps_filename, optimizer = HiGHS.Optimizer)
    h5 = H5File(h5_filename, "r")
    try
        cuts_lb = h5.get_array("cuts_lb")
        cuts_ub = h5.get_array("cuts_ub")
        cuts_lhs = h5.get_sparse("cuts_lhs")
        n_cuts = length(cuts_lb)
        @test n_cuts > 0
        @test n_cuts == length(cuts_ub)
        @test cuts_lhs.shape[1] == n_cuts
    finally
        h5.close()
    end
end
