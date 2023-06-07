using Pkg
Pkg.develop(path="/app")
Pkg.build("MIPLearn")

using MIPLearnT
runtests()
