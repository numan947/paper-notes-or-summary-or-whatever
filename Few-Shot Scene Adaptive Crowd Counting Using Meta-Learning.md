# [Few-Shot Scene Adaptive Crowd Counting Using Meta-Learning](https://arxiv.org/abs/2002.00264)

- given a target camera scene, goal is to adapt a model to this specific scene with only a few labeled images of that scene
- applicability in real world scenario, where we like to deploy a crowd counting model specially adapted to a target camera
- used learning-to-learn paradigm in the context of few-shot regime
- fast adaptation to the target scene
- proposed approach outperforms other alternatives in few shot scene adaptive crowd counting

## Introdcution
- Automated complex crowd scene understanding: surveillance, traffic monitoring,.... --> need crowd counting technique
- main limitation of exisiting technique --> hard to adapt to new crowd scene: require large number of training data->expensive and time consuming to obtain.
- new method: learns to adapt to a scene from very few labeled data
- Other methods: Crowd-counting --> supervised regression: model generates crowd density map for given image --> model learns to predict the density map of an input image given its ground-truth crowd density map as the label. Final count--> sum over the pixels in the estimated density map.
- Drawback: single model to be used in all unseen images, so training data need to be diverse.
- Recent work --> more effective to learn and deploy model specifically tuned for a particular scene.
- Once camera is installed images of that camera is constrained by camera parameters and 3D geometry of a specific scene. So, we don't need crowd counting to work for all arbitrary images but images from this specific camera.
- We want to adapt a model to a new camera.
- consider few shot scene adaptive crowd counting.
- During training: access to a set of training images from different scenes. During testing: new scene, new camera and a few (1~5) labeled images from this scene of this camera.
- During training learn optimal model parameters from multiple scene specific data by considering few labeled images per scene --> good initial point for adapting the model to a new scene specific camera.

![PROBLEM DEFINITION](images/kumarkm_1.png)


## Related Work

#### Crowd Counting
- Crowd counting can be grouped into: detection, regression or density-based methods *SEE PAPERS: 5,7,3,11,15,22,17*
- Two learning objectives for density estimation and crowd counting *SEE PAPER: 34*
- multi-column NN to handle input image at multiple scales to overcome the problem of scale variations *SEE PAPER 35*
- Encode local and global input image context to estimate the density map.
- This paper: uses another as backbone*SEE PAPER 16* 
- *SEE PAPER 4*: propose a non-CNN semi-supervised adaptation method by exploiting unlabeled data in the target domain (WHAT IS THIS?) need  corresponding samples that have common labels between the source and target domains
- propose to generate a large synthetic dataset and perform
domain adaptation to the real-world target domain *SEE PAPER 32* --> need prior knowledge of the distribution of the target domain

#### Few-Shot Learning
- Few shot learning: Learn a model from limited training examples for a task.
- Treat as meta-learning problem: NN as a learner to learn about a new task with just a few instances.
- Recent meta-learning: metric-based, model-based and optimization-based.
- metric based: learn a distance function to measure the similarity between data  points belonging to the same class.
- model based: deploy a model to store previously used training examples.
- optimization based: learns good initialization parameters based on learning from multiple tasks that favor fast adaptation on a new task.

## Few-shot Scene Adaptive Crowd Counting
#### Problem Setup

- Traditional ML: Given, $\mathcal{D} = \{\mathcal{D}^{train}, \mathcal{D}^{test}\}$, learn $f_{\theta} : \mathcal{x}\rightarrow\mathcal{y}$, where $\theta$ represent the parameters we learn by optimizing a defined loss function over the outputs in the train set.
- Few-shot meta-learning: trained on set of N tasks during meta-learning from $\mathcal{D}_{meta-train}$, where each task has its own train and tes set. $\mathcal{T}_i = \{\mathcal{D}_i^{train}, \mathcal{D}_i^{test}\}$ correspond to the $i^{th}$ task and both, $\mathcal{D}_i^{train}$ and $\mathcal{D}_i^{test}$ consists of labeled examples.
- Each camera scene is a task in this case. $i^{th}$ scene consists of  M labeled images. Randomly sample a small number $K \in \{1,5\}$ and $K\ll M$ labeled images for the $i^{th}$ scene. This reflects the real world scenario from having to learn from a few labeled data.
- Goal: learn the model in a way that it can adapt to a new scene using only a few training examples from the new scene during testing when available train set $\mathcal{D}_{new}^{train}$ consists of a few labeled images.
- Used $MAML$--> learns a  set of initial model parameters during the meta-learning stage which are used to initialize the model during meta-testing and is later fine-tuned on the few examples from a new target task.

#### Approach
- For a crowd-counting model, $f_\theta$, given input image $\mathcal{x}$, output: $f_\theta(\mathcal{x})$ is the crowd density map representing the density level at different spatial locations in the image. Summing over the entries in the generated density map ==  crowd count
- When adapting to a particular scene $\mathcal{T}_i$, the model parameters are updated using a few gradient steps to optimize the previously defined loss function.
- This can be thought as inner update during meta-learning and the optimization is expressed as:
  $\tilde{\theta_i} = \theta - \alpha\nabla_\theta\mathcal{L_\mathcal{T_i}}(f_\theta)$, where $\mathcal{L_\mathcal{T_i}}(f_\theta)$ = $\sum_{\mathcal{x^j, y^j}\in\mathcal{D^{train}_i}} ||f_\theta(\mathcal{x^j})-y^j||^2$ 

- Here $\mathcal{x^j}$ and $\mathcal{y^j}$ denotes the image and corresponding ground truth density map from the scene $\mathcal{T_i}$.
- During meta-learning phase, we learn the model parameters $\theta$ by optimizing $\mathcal{L_T}_i(f_{\tilde{\theta}_i})$ across N different training scenes.
![An overview of the approach](images/kumarkm_2.png)

#### Backbone Network Architecture

- Can be used with any backbone crowd counting network architecture.
- This paper uses CSRNet because of SOTA in crowd counting.
- CSRNet -> Feature Extractor + Density Map Estimator
- Feature Extractor: VGG-16 to extract a feature map of the input image. Used first 10 layers of VGG-16 as the feature extractor, output of the FE has 1/8 resolution of the input image.
- DME: consists of a series of dilated CNN layers to regress the output crowd density map for the given image.
- Used pretrained (on ImageNet) VGG-16 to initialize the Feature Extractor.
- Weights of the Density Map Estimator layers are initialized from Gaussian distribution with 0.01 std.
- Train on WorldExpo'10 --> baseline pretrained
- Susceptible when used for adaptation in few labeled data regime as it is not specifically designed to learn from few images ==> use this baseline pretrained network as the starting point for the meta-learning phase
- During meta-learning, fix the parameters of the feature extractor and train only density map estimator on different scene-specific data.


## Experiments

#### Datasets and Setup
- Required training images from multiple scenes
- WorldExpo'10 is the only such dataset with multiple scenes
- Also considered two other datasets: Mall, UCSD for cross-dataset testing.

**WordlExpo'10**
 - consists of 3980 labeled images from 1132 video sequences based on 108 different scenes
 - 103 for training and 5 for testing
 - image resolution: 576 X 720

**Mall**
 - 2000 images from the same camera setup inside a mall.
 - image resolution: 640 X 480
 - 800 training and 1200 testing

**UCSD**
 - 2000 images from same surveillance camera setup
 - captures pedestrian scene
 - relatively sparse crowd density (11~46)
 - resolution: 238 X 158
 - 800 for training and 1200 for testing

**Ground Truth Density Maps**
- All datasets have dot annotations, where each person in the image is annotated with a single point.
- Used Gaussian kernel to blur the point annotations in an image to create the ground-truth density map. ($\sigma$ = 3)

**Implementation:**
- PyTorch
- SGD with learning rate  = 0.001
- Adam with learning rate = 0.001

**Evaluation Metrics**
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MDE: Mean Deviation Error
- DIDN'T UNDERSTAND THE SUMMING OVER THE PIXEL TECHNIQUE

#### Baselines
**Baseline Pretrained:**
- standard crowd counting model trained in a standard supervised setting
- model parameters are trained from all images in the training set, once done, model is evaluated directly in new scene without adaptation.

**Baseline Fine-tuned:**
- Density maps are fine tuned for new scenes before evaluation to fine tune the model.

**Meta pre-trained:**
- without fine tuning on the target scene

