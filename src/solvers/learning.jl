global LearningSolver = PyNULL()

function __init_solvers_learning__()
    copy!(LearningSolver, pyimport("miplearn.solvers.learning").LearningSolver)
end

export LearningSolver
