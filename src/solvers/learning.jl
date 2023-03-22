#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

global LearningSolver = PyNULL()

function __init_solvers_learning__()
    copy!(LearningSolver, pyimport("miplearn.solvers.learning").LearningSolver)
end

export LearningSolver
