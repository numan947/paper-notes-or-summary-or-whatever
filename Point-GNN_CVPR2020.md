# [Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud](https://arxiv.org/pdf/2003.01251.pdf)

## Abstract
- GNN for detecting objects from LiDAR point cloud.
- Convert point cloud in a fixed radius near-neighbors graph.
- Point-GNN: predict category and shape of the object that each vertex in the graph belongs to.
- auto-registration mechanism to reduce translation variance
- design box merging and scoring operation to combine detections from multiple vertices accurately.
- leading accuracy using the point cloud alone and can surpass fusion-based algorithms
- Potential use of graph NN as a new approach for 3D object detection.