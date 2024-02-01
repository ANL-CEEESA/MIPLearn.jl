#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2024, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using SCIP

function gen_stab()
    np = pyimport("numpy")
    uniform = pyimport("scipy.stats").uniform
    randint = pyimport("scipy.stats").randint
    np.random.seed(42)
    gen = MaxWeightStableSetGenerator(
        w=uniform(10.0, scale=1.0),
        n=randint(low=50, high=51),
        p=uniform(loc=0.5, scale=0.0),
        fix_graph=true,
    )
    data = gen.generate(1)
    data_filenames = write_pkl_gz(data, "$BASEDIR/../fixtures", prefix="stab-n50-")
    collector = BasicCollector(write_mps=false)
    collector.collect(
        data_filenames,
        data -> build_stab_model_jump(data, optimizer=SCIP.Optimizer),
        progress=true,
        verbose=true,
    )
end

function test_cuts()
    data_filenames = ["$BASEDIR/../fixtures/stab-n50-0000$i.pkl.gz" for i in 0:0]
    clf = pyimport("sklearn.neighbors").KNeighborsClassifier(n_neighbors=1)
    extractor = H5FieldsExtractor(
        instance_fields=["static_var_obj_coeffs"],
    )
    comp = MemorizingCutsComponent(clf=clf, extractor=extractor)
    solver = LearningSolver(components=[comp])
    solver.fit(data_filenames)
    @show comp.n_features_
    @show comp.n_targets_
    stats = solver.optimize(
        data_filenames[1],
        data -> build_stab_model_jump(data, optimizer=SCIP.Optimizer),
    )
    @test stats["Cuts: AOT"] > 0
end
