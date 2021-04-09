using Conda
using PyCall

function install_miplearn()
    Conda.update()
    pip = joinpath(dirname(pyimport("sys").executable), "pip")
    isfile(pip) || error("$pip: invalid path")
    run(`$pip install miplearn==0.2.0.dev2`)
end

install_miplearn()
