#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2024, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using GLPK

function gen_tsp()
    np = pyimport("numpy")
    uniform = pyimport("scipy.stats").uniform
    randint = pyimport("scipy.stats").randint
    np.random.seed(42)

    gen = TravelingSalesmanGenerator(
        x = uniform(loc = 0.0, scale = 1000.0),
        y = uniform(loc = 0.0, scale = 1000.0),
        n = randint(low = 20, high = 21),
        gamma = uniform(loc = 1.0, scale = 0.25),
        fix_cities = true,
        round = true,
    )
    data = gen.generate(1)
    data_filenames = write_pkl_gz(data, "$BASEDIR/../fixtures", prefix = "tsp-n20-")
    collector = BasicCollector()
    collector.collect(
        data_filenames,
        data -> build_tsp_model_jump(data, optimizer = GLPK.Optimizer),
        progress = true,
        verbose = true,
    )
end

function test_lazy()
    data_filenames = ["$BASEDIR/../fixtures/tsp-n20-00000.pkl.gz"]
    clf = pyimport("sklearn.dummy").DummyClassifier()
    extractor = H5FieldsExtractor(instance_fields = ["static_var_obj_coeffs"])
    comp = MemorizingLazyComponent(clf = clf, extractor = extractor)
    solver = LearningSolver(components = [comp])
    solver.fit(data_filenames)
    model, stats = solver.optimize(
        data_filenames[1],
        data -> build_tsp_model_jump(data, optimizer = GLPK.Optimizer),
    )
    @test stats["Lazy Constraints: AOT"] > 0
end
