# [Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection](https://arxiv.org/pdf/1912.02424.pdf), Shifeng Zhang, et al.

## Abstract

- FPN and Focal Loss paved way for anchor-free (AF) detectors where anchor-based (AB) detectors dominated previously
- Pointed out that AF and AB differ in their definition of positive and negative training samples ==> performance gap.
-  If AB and AF define positive and negative training samples in the same way ==> no obvious difference between final performaces.
-  propose ATSS: Adaptive Training Sample Selection --> automatically selects positive and negative samples depending on statistical characteristics of object
-  ATSS improves performance of both AF and AB bridging their gap
-  Discuss the necessity of tiling multiple anchors per location on the image to detect objects
-  IMROVED SOTA BY A LARGE MARGIN to 50.7% AP w/o introducing any overhead

## Introduction
**Anchor Based Detectors**<br>
- Anchor-based Detectors -- 2 types:
- One stage methods (SSD & RetinaNet papers)
- Two stage methods (RCNN and FRCNN)
- tiles large number of preset anchors on the image, predicts category, refine the coordinates of these anchors several times, finally output these refined anchors
- Two stage methods refine several times more than one stage methods and thus, produce better results, but single-stage methods are more efficient

**Anchor Free Detectors**<br>
- Recent attention -- anchor-free methods because of FPN and Focal Loss
- two ways to find object
- Corner-based: locate several predefined or self-learned keypoints and then bound the spatial extent of objects: Keypoint-based methods (CornetNet, Bottom-up Object Detection)
- Center-based: use the center point or region of objects to define positives and then predict the four distances from positives to the object boundary. (FCOS, Foveabox)
- eliminates hyperparameters related to anchors, have similar performance with anchor-based detectors.


- keypoint-based detectors: follow standard keypoint estimation pipeline --> different from anchor based detectors
- center-based detectors: treat points as preset samples instead of anchor boxes --> similar to Anchor based detectors


**FCOS vs RetinaNet**: differs in 3 places<br>
- Number of anchors tiled per location: RetinaNet -- several per location, FCOS -- one per location
- Definition of positive and negative samples: RetinaNet -- resorts IoU for positives and negatives, FCOS -- uses spatial and scale constraints to select samples
- The regression starting status: RetinaNet -- regresses the object bbox from preset anchor box, FCOS -- locate object from the anchor point

**This Paper Contribution/Primary Works:** <br>
- investigate differences between anchor-based and anchor-free methods by ruling out the implementation inconsistencies between them
- Results indicated that the definition positive and negative examples is the main difference. If both select same positive and negative examples, no performance deviation is noted.
- Adaptive Training Sample Selection -- automatically selects positive and negative samples based on object characteristics -- bridges gap between anchor-based and anchor-free detectors.
- Series of experiments say that -- tiling multiple anchors per location is not necessary
1. Indicate the essential difference between AB and AF detectors.
2. Proposing ATSS
3. Demonstrating that tiling multiple anchors/location is unnecessary.
4. SOTA on MS COCO w/o any additional overhead.