# [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)

## Abstract
- method for training single DNN for detecting objects in images
- SSD discretizes the output space of bboxes into a set of default boxes over different aspect ratios and scales per feture map location.
- During prediction, network outputs scores for the presence of object at teach default box and produce adjustments to the box to better match the object's shape.
- Combines predictions from multiple feature maps from different levels (resolutions) to automatically handle objects of various sizes.
- eliminates region proposal and pixel resampling phase.
- Empirical experiments show that SSD is much faster while having competitive accuracy to RCNN and its variants.

## Introduction

- Region based Object detection: hypothesize bbox -- resample pixels for each box -- apply high quality classifier.
- Region based methods are computationally expensive for embedded systems and slow for real-time applications.
- Significant speed-up came at the cost of decreased detection accuracy.
- SSD: speed up the process by eliminating the region proposal and pixel sampling phase.
- Added series of improvements: using a small conv filter to predict object categories and offests in bboxes; using separate predictors for different aspect ratio detections; apply separate filter to multiple feature maps from later stages of a network to perform detection at multiple scales.

**Contributions:**<br>
- SSD



## Related Work

- Two methods for object detection: Sliding window based, region proposal based.
- Before CNNs, SOTA was: Deformable Part Model (DPM) for sliding window based models and Selective Search for region proposal based models.
- SPPNet, R-CNN, Fast-RCNN, Faster-RCNN
- SSD is similar to Faster-RCNN in a way that SSD and Faster-RCNN both uses a fixed set of default boxes for predictions.
- OverFeat, YOLO -- single shot detectors -- SSD can be reduced to these

## Conclusions
- Introduced SSD
- Uses multi-scale convolutional bounding box outputs attached to multiple feature maps at the end of the network.