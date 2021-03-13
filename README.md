BayesianExperiments.jl
====================

[![Latest Doc](https://img.shields.io/badge/docs-latest-blue.svg)](https://rakutentech.github.io/BayesianExperiments.jl/dev/)
[![codecov](https://codecov.io/gh/rakutentech/BayesianExperiments.jl/branch/main/graph/badge.svg?token=DOZ0HIW1V8)](https://codecov.io/gh/rakutentech/BayesianExperiments.jl)

A library for running Bayesian AB testing experiments.

Current features include:

- Conjugate prior models for distributions including Bernoulli, Normal, LogNormal, Exponential, etc.
- Basic models can be chained to model multiple steps process.
- Various stopping rules support: expected loss, probability to beat all.
- Support multiple experiment design including fixed horizon experiments, sequential test experiment and online learning.
- Efficient simulation tools to support power analysis.