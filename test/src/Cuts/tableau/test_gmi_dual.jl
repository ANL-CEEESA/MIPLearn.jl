#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2024, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using HiGHS

function test_cuts_tableau_gmi_dual()
    mps_filename = "$BASEDIR/../fixtures/bell5.mps.gz"
    h5_filename = "$BASEDIR/../fixtures/bell5.h5"
    stats = collect_gmi_dual(mps_filename, optimizer = HiGHS.Optimizer)
    h5 = H5File(h5_filename, "r")
    try
        cuts_basis_vars = h5.get_array("cuts_basis_vars")
        cuts_basis_sizes = h5.get_array("cuts_basis_sizes")
        cuts_rows = h5.get_array("cuts_rows")
        @test size(cuts_basis_vars) == (15, 402)
        @test size(cuts_basis_sizes) == (15,4)
        @test size(cuts_rows) == (15,)
    finally
        h5.close()
    end
end
