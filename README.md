<h1 align="center">MIPLearn.jl</h1>
<p align="center">
  <a href="https://github.com/ANL-CEEESA/MIPLearn.jl/actions">
    <img src="https://github.com/ANL-CEEESA/MIPLearn.jl/workflows/Test/badge.svg">
  </a>
  <a href="https://doi.org/10.5281/zenodo.4287567">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4287567.svg">
  </a>
  <a href="https://github.com/ANL-CEEESA/MIPLearn/discussions">
    <img src="https://img.shields.io/badge/GitHub-Discussions-%23fc4ebc" />
  </a>
</p>

**MIPLearn** is an extensible open-source framework for solving discrete optimization problems using a combination of Mixed-Integer Linear Programming (MIP) and Machine Learning (ML). See the [main repository](https://github.com/ANL-CEEESA/MIPLearn) for more information. This repository holds an experimental Julia interface for the package.

[miplearn]: https://github.com/ANL-CEEESA/MIPLearn

## 1. Usage

### 1.1 Installation

To use MIPLearn.jl, the first step is to [install the Julia programming language on your machine](https://julialang.org/). After Julia is installed, launch the Julia console, type `]` to switch to package manager mode, then run:

```
(@v1.6) pkg> add MIPLearn@0.2
```

This command should also automatically install all the required Python dependencies. To test that the package has been correctly installed, run (in package manager mode):

```
(@v1.6) pkg> test MIPLearn
```

If you find any issues installing the package, please do not hesitate to [open an issue](https://github.com/ANL-CEEESA/MIPLearn.jl/issues).


### 1.2 Describing instances

```julia
using JuMP
using MIPLearn

# Create problem data
weights = [1.0, 2.0, 3.0]
prices = [5.0, 6.0, 7.0]
capacity = 3.0

# Create standard JuMP model
model = Model()
n = length(weights)
@variable(model, x[1:n], Bin)
@objective(model, Max, sum(x[i] * prices[i] for i in 1:n))
@constraint(model, c1, sum(x[i] * weights[i] for i in 1:n) <= capacity)

# Add ML information
@feature(model, [5.0])
@feature(c1, [1.0, 2.0, 3.0])
@category(c1, "c1")
for i in 1:n
    @feature(x[i], [weights[i]; prices[i]])
    @category(x[i], "type-$i")
end

instance = JuMPInstance(model)
```

### 1.3 Solving instances and training


```julia
using MIPLearn
using Cbc

# Create training and test instances
training_instances = [...]
test_instances = [...]

# Create solver
solver = LearningSolver(Cbc.Optimizer)

# Solve training instances
for instance in train_instances
    solve!(solver, instance)
end

# Train ML models
fit!(solver, training_instances)

# Solve test instances
for instance in test_instances
    solve!(solver, instance)
end
```

### 1.4 Saving and loading solver state
```julia
using MIPLearn
using Cbc

# Solve training instances
training_instances = [...]
solver = LearningSolver(Cbc.Optimizer)
for instance in training_instances
    solve!(solver, instance)
end

# Train ML models
fit!(solver, training_instances)

# Save trained solver to disk
save!(solver, "solver.bin")

# Application restarts...

# Load trained solver from disk
solver = LearningSolver(Cbc.Optimizer)
load!(solver, "solver.bin")

# Solve additional instances
test_instances = [...]
for instance in test_instances
    solve!(solver, instance)
end

```

### 1.5 Solving training instances in parallel
```julia
using MIPLearn
using Cbc

# Solve training instances in parallel
training_instances = [...]
solver = LearningSolver(Cbc.Optimizer)
parallel_solve!(solver, training_instances, n_jobs=4)
fit!(solver, training_instances)

# Solve test instances in parallel
test_instances = [...]
parallel_solve!(solver, test_instances)
```

## 2. Customization

### 2.1 Selecting solver components

```julia
using MIPLearn

solver = LearningSolver(
  Cbc.Optimizer,
  components=[
    PrimalSolutionComponent(...),
    ObjectiveValueComponent(...),
  ]
)
```

### 2.2 Adjusting component aggressiveness

```julia
using MIPLearn

solver = LearningSolver(
  Cbc.Optimizer,
  components=[
    PrimalSolutionComponent(
      threshold=MinPrecisionThreshold(0.95),
    ),
  ]
)
```

### 2.3 Evaluating component performance

TODO

### 2.4 Using customized ML classifiers and regressors

TODO

## 3. Acknowledgments
* Based upon work supported by **Laboratory Directed Research and Development** (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357.
* Based upon work supported by the **U.S. Department of Energy Advanced Grid Modeling Program** under Grant DE-OE0000875.

## 4. Citing MIPLearn

If you use MIPLearn in your research (either the solver or the included problem generators), we kindly request that you cite the package as follows:

* **Alinson S. Xavier, Feng Qiu.** *MIPLearn: An Extensible Framework for Learning-Enhanced Optimization*. Zenodo (2020). DOI: [10.5281/zenodo.4287567](https://doi.org/10.5281/zenodo.4287567)

If you use MIPLearn in the field of power systems optimization, we kindly request that you cite the reference below, in which the main techniques implemented in MIPLearn were first developed:

* **Alinson S. Xavier, Feng Qiu, Shabbir Ahmed.** *Learning to Solve Large-Scale Unit Commitment Problems.* INFORMS Journal on Computing (2020). DOI: [10.1287/ijoc.2020.0976](https://doi.org/10.1287/ijoc.2020.0976)

## 5. License


Released under the modified BSD license. See `LICENSE` for more details.

