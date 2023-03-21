using PyCall

function test_problems_setcover()
    test_problems_setcover_generator()
    test_problems_setcover_model()
end

function test_problems_setcover_generator()
    np = pyimport("numpy")
    scipy_stats = pyimport("scipy.stats")
    randint = scipy_stats.randint
    uniform = scipy_stats.uniform

    np.random.seed(42)
    gen = SetCoverGenerator(
        n_elements = randint(low = 3, high = 4),
        n_sets = randint(low = 5, high = 6),
        costs = uniform(loc = 0.0, scale = 100.0),
        costs_jitter = uniform(loc = 0.95, scale = 0.10),
        density = uniform(loc = 0.5, scale = 0),
        K = uniform(loc = 25, scale = 0),
        fix_sets = false,
    )
    data = gen.generate(2)
    @test data[1].costs == [136.75, 86.17, 25.71, 27.31, 102.48]
    @test data[1].incidence_matrix == [
        1 0 1 0 1
        1 1 0 0 0
        1 0 0 1 1
    ]
    @test data[2].costs == [63.54, 76.6, 48.09, 74.1, 93.33]
    @test data[2].incidence_matrix == [
        1 1 0 1 1
        0 1 0 1 0
        0 1 1 0 0
    ]
end

function test_problems_setcover_model()
    data = SetCoverData(
        costs = [5, 10, 12, 6, 8],
        incidence_matrix = [
            1 0 0 1 0
            1 1 0 0 0
            0 0 1 1 1
        ],
    )

    h5 = H5File(tempname(), "w")
    model = build_setcover_model(data)
    model.extract_after_load(h5)
    model.optimize()
    model.extract_after_mip(h5)
    @test h5.get_scalar("mip_obj_value") == 11.0
    h5.close()
end
