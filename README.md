This repository contains accompanying code for the article introducing
Meta-Dataset,
[https://arxiv.org/abs/1903.03096](https://arxiv.org/abs/1903.03096).

This code is provided here in order to give more details on the implementation
of the data-providing pipeline, our back-bones and models, as well as the
experimental setting.

We are currently working on updating the code and adding user instructions to
facilitate reproducing the experiments and results of the article in new
execution environments, as well as design and run new experiments.

This is not an officially supported Google product.

## Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples

_Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Kelvin Xu,
Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol, Hugo
Larochelle_

Few-shot classification refers to learning a classifier for new classes given
only a few examples. While a plethora of models have emerged to tackle this
recently, we find the current procedure and datasets that are used to
systematically assess progress in this setting lacking. To address this, we
propose Meta-Dataset: a new benchmark for training and evaluating few-shot
classifiers that is large-scale, consists of multiple datasets, and presents
more natural and realistic tasks. The aim is to measure the ability of
state-of-the-art models to leverage diverse sources of data to achieve higher
generalization, and to evaluate that generalization ability in a more
challenging and realistic setting. We additionally measure robustness to
variations in the number of available examples and the number of classes.
Finally our extensive empirical evaluation leads us to identify weaknesses in
Prototypical Networks and MAML, two popular few-shot classification methods, and
to propose a new method, Proto-MAML, which achieves improved performance on our
benchmark.
