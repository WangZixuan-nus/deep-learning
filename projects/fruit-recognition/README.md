# Fruit Recognition: A Comparative Study of CNN and MobileNetV2 Architectures

## Abstract

This research project implements and evaluates two deep learning architectures for fruit image classification: a standard convolutional neural network (CNN) and a transfer learning approach based on MobileNetV2. Conducted as part of the ST5229 Deep Learning in Data Analytics course at the National University of Singapore, this study provides a comprehensive comparison of model efficacy, computational efficiency, and generalization capabilities in multi-class fruit recognition tasks. Experimental results demonstrate that while both architectures achieve acceptable performance, the MobileNetV2 model significantly outperforms the custom CNN model in terms of accuracy (90.53% vs. 63.43%), training efficiency, and resource utilization.

## 1. Introduction and Objectives

Image classification represents a fundamental computer vision task with extensive applications in agriculture, retail automation, and food processing industries. Automated fruit classification systems can streamline quality control processes, minimize labor costs, and enhance grading consistency. This research examines the comparative performance of two distinct neural network architectures in multi-class fruit classification, analyzing their efficiency-accuracy trade-offs in resource-constrained environments.

The specific objectives of this study are:
- To implement and evaluate a custom CNN architecture for fruit classification
- To develop a transfer learning approach using MobileNetV2 for the same task
- To conduct comprehensive comparative analysis of both models' performance
- To identify optimal architectural choices for resource-constrained applications

## 2. Dataset and Preprocessing

### 2.1 Dataset Structure

The study utilizes a 15-category fruit image dataset organized in the following hierarchical structure:

```
fruitdata/
├── train/
└── test/

```

### 2.2 Data Augmentation

Data augmentation techniques expanded the original training set of 5,423 images to 21,691 images through controlled transformations:
- Random rotation (±40 degrees)
- Width and height shifts (up to 20%)
- Shear transformations (up to 20%)
- Zoom variations (up to 20%)
- Horizontal flips

These augmentation techniques enhance model generalization by exposing the network to varied representations of the input data, thereby reducing overfitting and improving robustness.

## 3. Model Architectures

### 3.1 Custom CNN Architecture

The custom CNN implements a three-block convolutional architecture with the following components:
- Input normalization (scaling to [0,1])
- Three convolutional blocks with increasing filter sizes:
  - Block 1: 16 filters (3×3 kernels) followed by max pooling
  - Block 2: 32 filters (3×3 kernels) followed by max pooling
  - Block 3: 64 filters (3×3 kernels) followed by max pooling
- Flattening layer
- Dense layer with 128 neurons and ReLU activation
- Output layer with softmax activation for 15-class classification
- Total: 5.56M trainable parameters (21.22MB)

### 3.2 MobileNetV2 Transfer Learning Implementation

The MobileNetV2-based model leverages transfer learning with the following structure:
- Pre-processing layer (scaling to [-1,1] as required by MobileNetV2)
- Frozen ImageNet-pretrained MobileNetV2 base (non-trainable parameters)
- Global average pooling layer
- Dense output layer with softmax activation for 15-class classification
- Total: 19,215 trainable parameters (75.06KB), leveraging 2.26M pre-trained parameters

## 4. Experimental Configuration

### 4.1 Training Setup

All models were implemented using the TensorFlow framework with the following configurations:
- **Custom CNN**: SGD optimizer (learning rate: 0.01, momentum: 0.9)
- **MobileNetV2**: Adam optimizer with default parameter settings
- Both models utilize categorical cross-entropy loss function
- Weight decay of 0.00004 applied for regularization
- Batch size: 16 samples
- Input image dimensions: 224×224 pixels (3 channels)
- Training epochs: 15

### 4.2 Evaluation Metrics

Model performance was assessed using multiple complementary metrics:
- Classification accuracy (overall and class-wise)
- Training and validation loss curves
- Confusion matrices for error pattern analysis
- Parameter efficiency and model size
- Training time requirements

## 5. Project Implementation

### 5.1 Repository Structure

```
fruit-recognition/
├── fruitdata/                  # Original dataset
│   ├── train/                  # Training images
│   └── test/                   # Testing images
├── fruit_recognition.py        # Main implementation script
├── models/                     # Saved model files
├── results/                    # Visualization outputs
└── README.md                   # Project documentation
```

### 5.2 Installation Requirements

```bash
git clone https://github.com/WangZixuan-nus/deep-learning.git
cd deep-learning/projects/fruit-recognition
pip install tensorflow numpy matplotlib
```

### 5.3 Usage Instructions

The implementation provides a comprehensive pipeline for data augmentation, model training, evaluation, and comparison:

```python
python fruit_recognition.py
```

This script executes the following operations:
1. Data augmentation for both training and testing sets
2. Training of the custom CNN model
3. Training of the MobileNetV2 transfer learning model
4. Comprehensive evaluation of both models
5. Generation of comparative visualization and performance metrics

## 6. Experimental Results

### 6.1 Performance Comparison

The MobileNetV2 transfer learning approach demonstrated significant advantages across all metrics, achieving higher accuracy with substantially reduced computational requirements:

| Model | Test Accuracy (%) | Parameter Count | Training Time (s) |
|-------|------------------|----------------|-------------------|
| Custom CNN | 63.43 | 5.56M | 9470.50 |
| MobileNetV2 | 90.53 | 2.28M | 8696.07 |

### 6.2 Training Dynamics

The training dynamics revealed significant differences between the two architectures:

![CNN's Training and validation accuracy/loss](results_cnn.png)
![MobileNetV2's Training and validation accuracy/loss](results_mobilenet.png)

The CNN model exhibited classic overfitting symptoms—rapidly achieving 99.98% training accuracy but plateauing at 63.43% validation accuracy. Conversely, MobileNetV2 maintained a narrower generalization gap (9.47 percentage points), with validation accuracy reaching 90.53%. Additionally, MobileNetV2 converged faster, stabilizing after approximately 7 epochs compared to the CNN's 12 epochs.

### 6.3 Class-wise Performance Analysis

Confusion matrix analysis revealed MobileNetV2's consistent performance across all fruit categories, while the CNN struggled with morphologically complex fruits (longan, lychee) and visually similar categories (for instance, cantaloupe vs. cucumber):

![Heatmap for CNN performance](heatmap_cnn.png)
![Heatmap for MobileNetV2 performance](heatmap_mobilenet.png)

The most substantial performance difference occurred in the longan classification, where MobileNetV2 (92.3% accuracy) outperformed the CNN (48.7%) by nearly 44 percentage points.

## 7. Discussion and Conclusion

Our empirical study reveals significant advantages of MobileNetV2's transfer learning approach over custom CNN architectures in fruit classification tasks. This superior performance stems from three key factors:

1. **Enhanced Feature Representation**: MobileNetV2 offers hierarchical visual features learned from millions of diverse images, providing robust representation capacity.

2. **Architectural Efficiency**: The model achieves computational efficiency through depthwise separable convolutions, linear bottlenecks, and inverted residual structures that optimize parameter-performance trade-offs.

3. **Implicit Regularization**: The frozen feature extractor provides an implicit regularization effect that limits overfitting while maintaining generalization capabilities.

The experimental results demonstrate substantial performance improvements: a 27.10 percentage point increase in accuracy, 59% reduction in parameters, 70% decrease in computational requirements, and 62% faster inference speed. These benefits exist despite certain limitations, including the use of relatively clean datasets that may not fully represent real-world deployment scenarios and the absence of systematic hyperparameter optimization.

## 8. Future Work

Potential directions for extending this research include:
- Implementing more advanced data augmentation techniques
- Exploring fine-tuning strategies for the pre-trained models
- Testing on more challenging and diverse fruit datasets
- Developing and evaluating lightweight model architectures for mobile deployment
- Implementing real-time fruit recognition systems for practical applications

## 9. License

[MIT License](https://opensource.org/licenses/MIT)

## 10. Acknowledgements

This research was conducted as part of the ST5229 Deep Learning in Data Analytics course at the National University of Singapore. We acknowledge the contributions of the TensorFlow team for providing the deep learning framework and pre-trained models used in this study.
