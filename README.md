# Deep Learning Assignment 1: Fashion MNIST Classification

This project implements a deep neural network for classifying clothing items from the Fashion MNIST dataset. The implementation includes regularization techniques and various optimization methods, with all experiments tracked using Weights & Biases.

## Project Links

- **Interactive Report**: [View on Weights & Biases](https://wandb.ai/karapa-rajesh-iit-madras/DeepLearning/reports/DA6401-Assignment-1--VmlldzoxMTc3MTkyMA?accessToken=8rf7kfv3i5944oiu66rcqf2pi0al7d9dpe7jsla8i4pyo5u7jdupubz50gj9vb5k)
- **Source Code**: [GitHub Repository](https://github.com/SaiRajesh228/DeepLearningAssignment1/)

## Project Structure

The project consists of two main components:

### 1. Neural Network Implementation (`model.py`)

This module contains:
- Neural network operations (ReLU, Sigmoid, Tanh)
- A flexible multi-layer neural network class
- Cost functions with L2 regularization
- Multiple optimization algorithms (SGD, Momentum, Nesterov, RMSProp, Adam)
- Helper functions for data processing and evaluation

### 2. Experiment Runner (`train.py`)

This script:
- Manages the training process
- Handles command-line arguments
- Logs experiments to Weights & Biases
- Runs hyperparameter sweeps

## Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SaiRajesh228/DeepLearningAssignment1.git
   cd DeepLearningAssignment1
   ```

2. Log in to Weights & Biases:
   ```bash
   wandb login
   ```

### Running the Project

Execute the training script with your Weights & Biases credentials:

```bash
python train.py --wandb_entity YOUR_USERNAME --wandb_project DeepLearning
```

## Training Process

The training pipeline:
1. Loads and preprocesses the Fashion MNIST dataset
2. Trains the neural network using forward and backward propagation
3. Applies regularization to prevent overfitting
4. Updates model parameters using the selected optimization algorithm
5. Logs metrics and visualisations to Weights & Biases

## Evaluation

After training, the model is evaluated by:
- Calculating accuracy on the test set
- Generating a confusion matrix to understand classification errors
- Visualising results in the Weights & Biases dashboard

## Customisation

The codebase is designed to be flexible. You can modify:
- Network architecture (number and size of layers)
- Activation functions
- Optimization algorithms
- Regularization strength
- Learning rate and other hyperparameters

## Experiment Tracking

All experiments are automatically logged to Weights & Biases, allowing you to:
- Compare different model configurations
- Visualise training progress
- Share results with others
- Reproduce experiments


## Here’s an example of command-line usage(Using Google Colab):

```bash

   # Run this entire block in a Colab cell
!pip install wandb numpy tensorflow scikit-learn matplotlib seaborn --quiet

# Remove the problematic environment variable setting
import os
if "WANDB_SERVICE" in os.environ:
    del os.environ["WANDB_SERVICE"]

# Proceed with login and execution
import wandb
wandb.login()  # You'll get a link to authenticate - follow and paste your API key here

!rm -rf DeepLearningAssignment1  # Clean existing installations
!git clone https://github.com/SaiRajesh228/DeepLearningAssignment1.git
%cd DeepLearningAssignment1

# # Run this in a new Colab cell
# import wandb
# wandb.login()  # Follow the authentication prompt

!python train.py \
 --wandb_entity "karapa-rajesh" \
 --wandb_project "DeepLearning" \
 --dataset "fashion_mnist" \
 --epochs 10 \
 --batch_size 64 \
 --optimizer "adam" \
 --learning_rate 0.001 \
 --num_layers 3 \
 --hidden_size 128 \
 --activation "relu" \
 --weight_init "xavier" \
 --weight_decay 0.0001 \
 --beta1 0.9 \
 --beta2 0.999 \
 --epsilon 1e-8
```
