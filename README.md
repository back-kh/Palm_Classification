# Isolated Palm Leaf Classification Benchmark

## Overview

Welcome to the Isolated Palm Leaf Classification Benchmark! This project aims to improve the classification of isolated palm leaf images using various Convolutional Neural Network (CNN) architectures. Leveraging the strengths of models like VGG, ResNet, Efficient ViT, Swin, and CvT, we present a framework for enhancing classification accuracy and efficiency.

## Project Goals

- **Develop a Benchmark:** Establish a standard benchmark for isolated palm leaf classification.
- **Compare Architectures:** Evaluate and compare the performance of different neural network architectures.
- **Improve Classification:** Utilize advanced techniques to enhance classification accuracy.

## Datasets

We employ a curated dataset of isolated palm leaf images, consisting of multiple species. The dataset is split into training, validation, and test sets.

## Models

1. **VGG (Visual Geometry Group)**
   - **Description:** VGG is known for its simplicity and depth, utilizing small convolutional filters to capture intricate details. Its architecture allows for a straightforward approach to feature extraction.
   - **Use Case:** Best for scenarios where interpretability and feature visualization are essential.

2. **ResNet (Residual Networks)**
   - **Description:** ResNet introduces skip connections, enabling the training of very deep networks without the vanishing gradient problem. This architecture allows for more effective feature learning.
   - **Use Case:** Ideal for leveraging deeper architectures to significantly improve classification accuracy.

3. **Efficient ViT (Vision Transformer)**
   - **Description:** Efficient ViT combines the local feature extraction of convolutional layers with the global context capabilities of transformers, optimizing both computational efficiency and performance.
   - **Use Case:** Particularly effective for datasets with high variability and complexity.

4. **Swin Transformer**
   - **Description:** Swin Transformer employs a hierarchical architecture that processes images at various scales, effectively capturing local and global features.
   - **Use Case:** Suitable for tasks requiring high-resolution inputs, making it versatile across different image sizes.

5. **CvT (Convolutional Vision Transformer)**
   - **Description:** CvT merges convolutional layers with transformer architectures, improving local feature extraction while maintaining global understanding, resulting in better performance in complex classification tasks.
   - **Use Case:** Excels in scenarios that require a balance between local detail and global context.


## Evaluation Metrics

- **Accuracy:** Proportion of correctly classified images.
- **F1 Score:** Harmonic mean of precision and recall.

## Installation

To set up the environment, clone this repository and install the required dependencies:

## Citations
@inproceedings{thuon2022improving,
  title={Improving isolated glyph classification task for palm leaf manuscripts},
  author={Thuon, Nimol and Du, Jun and Zhang, Jianshu},
  booktitle={International Conference on Frontiers in Handwriting Recognition},
  pages={65--79},
  year={2022},
  organization={Springer}
}
