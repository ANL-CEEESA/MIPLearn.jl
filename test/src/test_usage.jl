
function test_usage()
    LogisticRegression = pyimport("sklearn.linear_model").LogisticRegression

    @debug "Generating data files..."
    dirname = tempdir()
    data = [fixture_setcover_data()]
    data_filenames = write_pkl_gz(data, dirname)
    h5_filenames = ["$(f).h5" for f in data_filenames]

    @debug "Setting up LearningSolver..."
    solver = LearningSolver(
        components = [
            IndependentVarsPrimalComponent(
                base_clf = SingleClassFix(
                    MinProbabilityClassifier(
                        base_clf = LogisticRegression(),
                        thresholds = [0.95, 0.95],
                    ),
                ),
                extractor = AlvLouWeh2017Extractor(),
                action = SetWarmStart(),
            ),
        ],
    )

    @debug "Collecting training data..."
    bc = BasicCollector()
    bc.collect(data_filenames, build_setcover_model)

    @debug "Training models..."
    solver.fit(data_filenames)

    @debug "Solving model..."
    solver.optimize(data_filenames[1], build_setcover_model)

    @debug "Checking solution..."
    h5 = H5File(h5_filenames[1])
    @test h5.get_scalar("mip_obj_value") == 11.0
end
