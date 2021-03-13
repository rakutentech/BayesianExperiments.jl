# BayesianExperiments.jl

[![Latest Doc](https://img.shields.io/badge/docs-latest-blue.svg)][1]
[![codecov](https://codecov.io/gh/rakutentech/BayesianExperiments.jl/branch/main/graph/badge.svg?token=DOZ0HIW1V8)](https://codecov.io/gh/rakutentech/BayesianExperiments.jl)

Tools for running Bayesian AB test experiments.

Current features include:

- Conjugate prior models for distributions including Bernoulli, Normal, LogNormal, Exponential, etc.
- Basic models can be chained to model multiple steps process.
- Various stopping rules support: expected loss, probability to beat all.
- Support multiple experiment design including fixed horizon experiments, sequential test experiment and online learning.
- Efficient simulation tools to support power analysis.

## Documentation

For usage instructions and tutorials, see [documentation][1].

[1]: https://rakutentech.github.io/BayesianExperiments.jl/dev/