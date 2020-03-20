# [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489)

## Abstract

- Increasing batch size during training is equivalent to decreasing learning rate for SGD, SGD+momentum, Nesterov momentum, and Adam.
- Reaches equivalent test accuracies after same number of training epochs but with fewer parameter updates which leads to greater parallelism and shorter training times.
- Further reduce parameter updates by increasing learning rate, ϵ and scaling the batch size B ∝ ϵ
- Can also increase momentum coefficient m, and scale B ∝ 1/(1-m) -- may lead to slightly reduced test accuracy.
- trained ResNet-50 on ImageNet to 76.1% accuracy in under 30 minutes.

## Introduction | Conclusion

- SGD - dominant optimization algorithm for deep learning - slow: each parameter update takes a small step towards the objective
- Large batch sizes -- reduces training time but test set accuracy falls.
- Decaying learning rate simulated annealing ⇒ reduce random fluctuations in SGD dynamics.
- Instead of decreasing LR, increase batch size: when learning rate drops by a factor of α, increase the batch size by a factor of α -- achieves almost identical model performance but significantly lower number of parameter upgrades.
- Further reduce number of parameter updates b increasing learning rate and scaling B ∝ ϵ, also increase momentum coefficient m, and scale B ∝ 1/(1-m)

