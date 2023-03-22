
global FeaturesExtractor = PyNULL()
global AlvLouWeh2017Extractor = PyNULL()
global DummyExtractor = PyNULL()
global H5FieldsExtractor = PyNULL()

function __init_extractors__()
    copy!(FeaturesExtractor, pyimport("miplearn.extractors.abstract").FeaturesExtractor)
    copy!(
        AlvLouWeh2017Extractor,
        pyimport("miplearn.extractors.AlvLouWeh2017").AlvLouWeh2017Extractor,
    )
    copy!(DummyExtractor, pyimport("miplearn.extractors.dummy").DummyExtractor)
    copy!(H5FieldsExtractor, pyimport("miplearn.extractors.fields").H5FieldsExtractor)

end

export FeaturesExtractor, AlvLouWeh2017Extractor, DummyExtractor, H5FieldsExtractor
