JULIA := julia --color=yes
JULIA_SYSIMAGE_ARGS := $(JULIA_ARGS) --sysimage build/sysimage.so

all: test

build/sysimage.so: Manifest.toml Project.toml
	mkdir -p build
	$(JULIA) --project=test test/sysimage.jl

test: build/sysimage.so
	$(JULIA) --sysimage build/sysimage.so --project=test test/runtests.jl

.PHONY: test test-python test-julia test-watch docs install
