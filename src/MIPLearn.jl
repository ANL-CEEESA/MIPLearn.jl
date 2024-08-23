#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

module MIPLearn

using PyCall
using SparseArrays
using PrecompileTools: @setup_workload, @compile_workload


include("collectors.jl")
include("components.jl")
include("extractors.jl")
include("io.jl")
include("problems/setcover.jl")
include("problems/stab.jl")
include("problems/tsp.jl")
include("solvers/jump.jl")
include("solvers/learning.jl")

function __init__()
    __init_collectors__()
    __init_components__()
    __init_extractors__()
    __init_io__()
    __init_problems_setcover__()
    __init_problems_stab__()
    __init_problems_tsp__()
    __init_solvers_jump__()
    __init_solvers_learning__()
end

include("BB/BB.jl")
include("Cuts/Cuts.jl")

# Precompilation
# =============================================================================

function __precompile_cuts__()
    function build_model(mps_filename)
        model = read_from_file(mps_filename)
        set_optimizer(model, SCIP.Optimizer)
        return JumpModel(model)
    end
    BASEDIR = dirname(@__FILE__)
    mps_filename = "$BASEDIR/../test/fixtures/bell5.mps.gz"
    h5_filename = "$BASEDIR/../test/fixtures/bell5.h5"
    collect_gmi_dual(
        mps_filename;
        optimizer=HiGHS.Optimizer,
        max_rounds = 10,
        max_cuts_per_round = 500,
    )
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
end

@setup_workload begin
    using SCIP
    using HiGHS
    using MIPLearn.Cuts
    using PrecompileTools: @setup_workload, @compile_workload

    __init__()
    Cuts.__init__()

    @compile_workload begin
        __precompile_cuts__()
    end
end

end # module
