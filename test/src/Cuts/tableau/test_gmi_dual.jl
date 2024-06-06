#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2024, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using SCIP
using HiGHS

function test_cuts_tableau_gmi_dual_collect()
    mps_filename = "$BASEDIR/../fixtures/bell5.mps.gz"
    h5_filename = "$BASEDIR/../fixtures/bell5.h5"
    stats = collect_gmi_dual(mps_filename, optimizer = HiGHS.Optimizer)
    h5 = H5File(h5_filename, "r")
    try
        cuts_basis_vars = h5.get_array("cuts_basis_vars")
        cuts_basis_sizes = h5.get_array("cuts_basis_sizes")
        cuts_rows = h5.get_array("cuts_rows")
        @test size(cuts_basis_vars) == (15, 402)
        @test size(cuts_basis_sizes) == (15, 4)
        @test size(cuts_rows) == (15,)
    finally
        h5.close()
    end
end

function test_cuts_tableau_gmi_dual_usage()
    function build_model(mps_filename)
        model = read_from_file(mps_filename)
        set_optimizer(model, SCIP.Optimizer)
        return JumpModel(model)
    end

    mps_filename = "$BASEDIR/../fixtures/bell5.mps.gz"
    h5_filename = "$BASEDIR/../fixtures/bell5.h5"
    # rm(h5_filename, force=true)

    # # Run basic collector
    # bc = BasicCollector(write_mps = false, skip_lp = true)
    # bc.collect([mps_filename], build_model)

    # # Run dual GMI collector
    # @info "Running dual GMI collector..."
    # collect_gmi_dual(mps_filename, optimizer = HiGHS.Optimizer)

    # # Test expert component
    # solver = LearningSolver(
    #     components = [
    #         ExpertPrimalComponent(action = SetWarmStart()),
    #         ExpertDualGmiComponent(),
    #     ],
    #     skip_lp = true,
    # )
    # solver.optimize(mps_filename, build_model)

    # Test kNN component
    knn = KnnDualGmiComponent(
        extractor = H5FieldsExtractor(instance_fields = ["static_var_obj_coeffs"]),
        k = 2,
    )
    knn.fit([h5_filename, h5_filename])
    solver = LearningSolver(
        components = [
            ExpertPrimalComponent(action = SetWarmStart()),
            knn,
        ],
        skip_lp = true,
    )
    solver.optimize(mps_filename, build_model)

    return
end
