# BayesianExperiments.jl

[![Latest Doc](https://img.shields.io/badge/docs-latest-blue.svg)][2]
[![Stable Doc](https://img.shields.io/badge/docs-stable-blue.svg)][1]
[![codecov](https://codecov.io/gh/rakutentech/BayesianExperiments.jl/branch/main/graph/badge.svg?token=DOZ0HIW1V8)](https://codecov.io/gh/rakutentech/BayesianExperiments.jl)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rakutentech/BayesianExperiments.jl/main?filepath=examples)

`BayesianExperiments.jl` provides a toolbox for running various types of Bayesian AB testing experiments.

Current features include:

- Hypothesis testing with Bayes factor. Support the effect size model with Normal distribution prior and JZS prior.
- Bayesian decision making with conjugate prior models. Support expected loss and probability to beat all as the stopping rule.
- Flexible experiment design for both fixed horizon experiments and sequential test experiment.
- Efficient simulation tools to support power analysis and sensitivity analysis.

## Documentation and Examples

For usage instructions and tutorials, see [documentation][1].

For detailed discussions on many topics in the field, see the Jupyter notebooks in the `examples` folder:

- [Sequential Experiment with Two Models](examples/sequential_testing_conjugate_models.ipynb)
- [Type S Error in Fixed Horizon and Sequential Test Experiment](examples/type_s_error.ipynb)
- [Bayes Factor Experiment with Optional Stopping](examples/bayes_factor_optional_stopping.ipynb)

Or you can go to [binder](https://mybinder.org/v2/gh/rakutentech/BayesianExperiments.jl/main?filepath=examples) to directly play with the Jupyter notebooks.

[1]: https://rakutentech.github.io/BayesianExperiments.jl/stable/
[2]: https://rakutentech.github.io/BayesianExperiments.jl/latest/

## Related Projects

Open source projects in R related to our project:

- [easystats/bayestestR](https://github.com/easystats/bayestestR/)
- [FrankPortman/bayesAB](https://github.com/FrankPortman/bayesAB)
- [richarddmorey/BayesFactor](https://github.com/richarddmorey/BayesFactor)
