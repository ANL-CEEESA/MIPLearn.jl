#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

VERSION := 0.2

test:
	./juliaw test/runtests.jl

format:
	cd deps/formatter; ../../juliaw format.jl

.PHONY: docs test format
