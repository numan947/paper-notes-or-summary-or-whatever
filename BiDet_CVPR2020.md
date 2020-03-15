# [BiDet: An Efficient Binarized Object Detector](https://arxiv.org/pdf/2003.03961.pdf), Ziwei Wang

## Abstract

- Efficient Object detection
- CNN binarization methods: directly learn the weights and activations in one or two stage detectors with constrained representational capacity --> information redundancy causes numerous false positives and degrades the performance significantly
- BiDet: fully uses the representational capacity of binary NNs for object detection by redundancy removal --> enhanced detection precision, alleviated FPs
- Generalized Information Bottleneck principle to object detection: amount of information in high level feature maps is constrained and the mutual information between the feature maps and object detection is maximized
- Learned sparse object priors to make posterior concentrate on informative detection prediction with FP elimination
- SOTA: BNNs

## Introduction

- CNN-based Object Detection requires massive computation and storage resources for ideal performance -- limits deployment on mobiel devices -- need lightweight architectures

**Proposed Compression Methods:**<br>
1. Prunning (Channel Prunning, Importance Estimation, Variational Conv)
2. Low-rank decomposition (Compression ...)
3. Quantization
4. Knowledge distillation
5. Architecture Design
6. Architecture Search(37,43)

- Quantization reduces bitwidth of the network parameters and activations for efficient inference
- Binarizing weights and activations an decrease storage cost by 32X and computation cost by 64X
- BNNs cause numerous FPs for the information redundancy

**This Paper:**<br>
- BiDet for learning binarized NNs including backbone and detection for efficient Object Detection.
- Uses BNNs's representational capacity by removing redundancy which lead to enhanced detection with FP elimination.
- Information Bottleneck principle on BNNs
- Simultaneously limit the amount of information in high level FMs and maximize the mutual information between object detection and the learned FMs.
- Use the large sparse object priors to concentrate posteriors on informative prediction and uninformative FPs are eliminated
- outperforms SOTA BNNs for Object Detection

**Contribution:**<br>
1. First BNN containing the backbone and detection parts for efficient object detection
2. Use IB principle for redundancy removal to fully utilize the capacity of BNNs and learn sparse object priors to concentrate posteriors on informative detection prediction for enhanced detection accuracy and lower FPs.
3. Evaluated on PASCAL VOC and COCO: SOTA BNN based object detector

## Conclusion
- Proposed BiDet -- binarized NN learning method.
- BiDet removes redundant information via information bottleneck principle to fully use the representational capacity of BNNs and enforce posteriors to be concentrated on informative prediction for FP elimination which leads to significan precision enhancement.
- SOTA for BNN object detection