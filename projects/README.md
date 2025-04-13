# Fruit Recognition Project

## Overview
This project implements a fruit image classification system using deep learning techniques. It compares the performance of a custom CNN model against a transfer learning approach with MobileNetV2 on a fruit image dataset.

## Features
- Data augmentation to enhance training dataset diversity
- Implementation of a custom CNN architecture
- Implementation of transfer learning using MobileNetV2
- Comprehensive model evaluation with confusion matrices
- Performance comparison between models

## Dataset
The project uses a fruit image dataset organized in the following structure:
```
fruitdata/
├── train/
└── test/
```

## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/WangZixuan-nus/deep-learning.git
cd deep-learning/projects/fruit-recognition
pip install tensorflow numpy matplotlib
```

## Usage
Run the main script to:
1. Augment the dataset
2. Train both CNN and MobileNetV2 models
3. Evaluate and compare model performance

```python
python fruit_recognition.py
```

## Model Architecture

### Custom CNN
The custom CNN model consists of:
- 3 convolutional blocks with increasing filter sizes (16, 32, 64)
- Max pooling layers after each convolutional layer
- A fully connected layer with 128 units
- Output layer with softmax activation for multi-class classification

### MobileNetV2 Transfer Learning
The MobileNetV2-based model:
- Uses pre-trained MobileNetV2 (trained on ImageNet) as the base model
- Adds a global average pooling layer
- Includes a custom output layer for fruit classification

## Results
The models are evaluated based on:
- Training and validation accuracy
- Training and validation loss
- Confusion matrix visualization
- Direct accuracy comparison

Results are saved in the `/results` directory and include:
- Training history plots
- Confusion matrix heatmaps
- Model comparison bar charts

## Project Structure
```
fruit-recognition/
├── fruitdata/                  # Original dataset
│   ├── train/                  # Training images
│   └── test/                   # Testing images
├── fruit_recognition.py        # Main implementation
├── models/                     # Saved model files
├── results/                    # Visualization outputs
└── README.md                   # This file
```

## Model Performance
The project compares two models:
1. Custom CNN: A lightweight model built from scratch
2. MobileNetV2: A transfer learning approach using a pre-trained model

Performance metrics include accuracy, loss, and class-specific recognition rates.


## License
[MIT License](https://opensource.org/licenses/MIT)

## Acknowledgements
- This project is part of the coursework for ST5229 Deep Learning in Data Analytics at NUS
