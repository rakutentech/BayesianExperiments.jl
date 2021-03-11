# Bayesian experiments

A library for running Bayesian AB testing.

Current features includes:

- Conjugate prior models for distributions including Bernoulli, Normal, LogNormal, Exponential, etc.
- Basic models can be chained to model multiple steps process.
- Various stopping rules support: expected loss, probability to beat all.
- Support multiple experiment design including fixed horizon experiments, sequential experiment and online learning.
- Efficient Simulation tools to support power analysis.
