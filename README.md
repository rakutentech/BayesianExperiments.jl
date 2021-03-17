# BayesianExperiments.jl

[![Latest Doc](https://img.shields.io/badge/docs-latest-blue.svg)][1]
[![codecov](https://codecov.io/gh/rakutentech/BayesianExperiments.jl/branch/main/graph/badge.svg?token=DOZ0HIW1V8)](https://codecov.io/gh/rakutentech/BayesianExperiments.jl)

`BayesianExperiments.jl` provides a toolbox for running various types of Bayesian AB test experiments.

Current features include:

- Hypothesis testing with Bayes factor. Support the effect size model with unit information prior and JZS prior.
- Bayesian decision making with conjugate prior models. Support expected loss and probability to beat all as the stopping rule.
- Flexible experiment design for both fixed horizon experiments and sequential test experiment.
- Efficient simulation tools to support power analysis and sensitivity analysis.

## Documentation and Examples

For usage instructions and tutorials, see [documentation][1].

For detailed discussions on many topics in the filed, see the Jupyter notebooks in the `examples` folder:

- [Sequential Experiemnt with Two Models](examples/sequential_experiment_two_models.ipynb)
- [Type S Error of Fixed Horizon and Sequential Test Experiement](examples/fixed_vs_sequentail_type_s_error.ipynb)

[1]: https://rakutentech.github.io/BayesianExperiments.jl/dev/

## Related Projects

Open source projects in R related to our project:

- [`bayesTestR`](https://easystats.github.io/bayestestR/)
- [`bayesAB`](https://github.com/FrankPortman/bayesAB)
- [`BayesFactor`](https://github.com/richarddmorey/BayesFactor)