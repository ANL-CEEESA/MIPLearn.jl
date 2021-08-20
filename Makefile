#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

JULIA := julia --color=yes --project=@.
VERSION := 0.2

build/sysimage.so: src/utils/sysimage.jl Project.toml
	julia --project=. -e "using Pkg; Pkg.instantiate()"
	$(JULIA) src/utils/sysimage.jl test/runtests.jl

clean:
	rm -rfv build/*

test: build/sysimage.so
	$(JULIA) --sysimage build/sysimage.so test/runtests.jl

format:
	julia -e 'using JuliaFormatter; format(["src", "test", "benchmark"], verbose=true);'

install-deps:
	julia -e 'using Pkg; Pkg.add(PackageSpec(name="JuliaFormatter", version="0.14.4"))'

.PHONY: docs test format install-deps
