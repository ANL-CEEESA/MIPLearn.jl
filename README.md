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


### 1.3 Solving instances and training

### 1.4 Saving and loading solver state

### 1.5 Solving training instances in parallel

## 2. Customization

### 2.1 Selecting solver components

### 2.2 Adjusting component aggresiveness

### 2.3 Evaluating component performance

### 2.4 Using customized ML classifiers and regressors

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

