#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2024, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using PyCall
using SCIP

function test_problems_stab()
    nx = pyimport("networkx")
    data = MaxWeightStableSetData(
        graph=nx.gnp_random_graph(25, 0.5, seed=42),
        weights=repeat([1.0], 25),
    )
    h5 = H5File(tempname(), "w")
    model = build_stab_model_jump(data, optimizer=SCIP.Optimizer)
    model.extract_after_load(h5)
    model.optimize()
    model.extract_after_mip(h5)
    @test h5.get_scalar("mip_obj_value") == -6
    @test h5.get_scalar("mip_cuts")[1:20] == "[[0,8,11,13],[0,8,13"
    h5.close()
end
