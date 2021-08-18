# UnitCommitment.jl: Optimization Package for Security-Constrained Unit Commitment
# Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
# Released under the modified BSD license. See COPYING.md for more details.

JULIA := julia --color=yes --project=@.
VERSION := 0.2

build/sysimage.so: src/utils/sysimage.jl Project.toml Manifest.toml
	mkdir -p build
	$(JULIA) --trace-compile=build/precompile.jl test/runtests.jl
	$(JULIA) src/utils/sysimage.jl

clean:
	rm -rf build/*

test: build/sysimage.so
	$(JULIA) --sysimage build/sysimage.so test/runtests.jl

format:
	julia -e 'using JuliaFormatter; format(["src", "test", "benchmark"], verbose=true);'

install-deps:
	julia -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaFormatter", version="0.14.4"))'

.PHONY: docs test format install-deps
