using Conda
ENV["PATH"] *= ";" * joinpath(Conda.ROOTENV, "Scripts")
ENV["PATH"] *= ";" * joinpath(Conda.ROOTENV, "Library", "bin")
pip = joinpath(Conda.ROOTENV, "Scripts", "pip")
run(`$pip install miplearn==0.1.0`)

