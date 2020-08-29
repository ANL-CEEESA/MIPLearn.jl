using Conda

function install_miplearn()
    paths = [joinpath(Conda.ROOTENV, "Scripts"),
             joinpath(Conda.ROOTENV, "Library", "bin"),
             joinpath(Conda.ROOTENV, "bin")]

    pip_found = false
    for p in paths
        if isfile("$p/pip3")
            run(`$p/pip3 install miplearn==0.1.0`)
            pip_found = true
            break
        end
    end

    pip_found || error("Could not find pip")
end

Conda.update()
install_miplearn()
