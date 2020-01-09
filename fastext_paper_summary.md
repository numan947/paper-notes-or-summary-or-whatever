# [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)

- paper introduces a simple and efficient baseline for text classification
- often on par with deep learning classifiers in terms of accuracy and many times faster for training and evaluation.
- train on 10B words in less than 10 minutes using standard multi-core cpu and classify 500K sentences among 312K classes in less than a minute.

## Introduction
Text classification applications:
- web search
- information retrieval
- ranking
- document classification

NN models -- provide good performances but relatively slow in train and test time -- limits usage in very large datasets.

Linear classifiers -- simple but can obtain SOTA if right features are selected. These scale to very large corpus.

*This paper explore ways to scale linear classifiers to very large corpus in context of text classification.*
- inspired by recent work in word representation learning
- shows that linear models with a rank constraint and fast loss approximation can train faster while achieving performance on par with SOTA.
- Evaluate *fastText* on tag prediction and sentiment analysis.

## Model Architecture 