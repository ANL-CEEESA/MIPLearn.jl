
global MinProbabilityClassifier = PyNULL()
global SingleClassFix = PyNULL()
global PrimalComponentAction = PyNULL()
global SetWarmStart = PyNULL()
global FixVariables = PyNULL()
global EnforceProximity = PyNULL()
global ExpertPrimalComponent = PyNULL()
global IndependentVarsPrimalComponent = PyNULL()
global JointVarsPrimalComponent = PyNULL()
global SolutionConstructor = PyNULL()
global MemorizingPrimalComponent = PyNULL()
global SelectTopSolutions = PyNULL()
global MergeTopSolutions = PyNULL()

function __init_components__()
    copy!(
        MinProbabilityClassifier,
        pyimport("miplearn.classifiers.minprob").MinProbabilityClassifier,
    )
    copy!(SingleClassFix, pyimport("miplearn.classifiers.singleclass").SingleClassFix)
    copy!(
        PrimalComponentAction,
        pyimport("miplearn.components.primal.actions").PrimalComponentAction,
    )
    copy!(SetWarmStart, pyimport("miplearn.components.primal.actions").SetWarmStart)
    copy!(FixVariables, pyimport("miplearn.components.primal.actions").FixVariables)
    copy!(EnforceProximity, pyimport("miplearn.components.primal.actions").EnforceProximity)
    copy!(
        ExpertPrimalComponent,
        pyimport("miplearn.components.primal.expert").ExpertPrimalComponent,
    )
    copy!(
        IndependentVarsPrimalComponent,
        pyimport("miplearn.components.primal.indep").IndependentVarsPrimalComponent,
    )
    copy!(
        JointVarsPrimalComponent,
        pyimport("miplearn.components.primal.joint").JointVarsPrimalComponent,
    )
    copy!(
        SolutionConstructor,
        pyimport("miplearn.components.primal.mem").SolutionConstructor,
    )
    copy!(
        MemorizingPrimalComponent,
        pyimport("miplearn.components.primal.mem").MemorizingPrimalComponent,
    )
    copy!(SelectTopSolutions, pyimport("miplearn.components.primal.mem").SelectTopSolutions)
    copy!(MergeTopSolutions, pyimport("miplearn.components.primal.mem").MergeTopSolutions)
end

export MinProbabilityClassifier,
    SingleClassFix,
    PrimalComponentAction,
    SetWarmStart,
    FixVariables,
    EnforceProximity,
    ExpertPrimalComponent,
    IndependentVarsPrimalComponent,
    JointVarsPrimalComponent,
    SolutionConstructor,
    MemorizingPrimalComponent,
    SelectTopSolutions,
    MergeTopSolutions
