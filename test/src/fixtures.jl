function fixture_setcover_data()
    return SetCoverData(
        costs = [5, 10, 12, 6, 8],
        incidence_matrix = [
            1 0 0 1 0
            1 1 0 0 0
            0 0 1 1 1
        ],
    )
end

function fixture_setcover_model()
    return build_setcover_model(fixture_setcover_data())
end
