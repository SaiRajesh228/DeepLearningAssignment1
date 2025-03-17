## Report : https://wandb.ai/karapa-rajesh-iit-madras/DeepLearning/reports/DA6401-Assignment-1--VmlldzoxMTc3MTkyMA?accessToken=8rf7kfv3i5944oiu66rcqf2pi0al7d9dpe7jsla8i4pyo5u7jdupubz50gj9vb5k

## Github Repo : https://github.com/SaiRajesh228/DeepLearningAssignment1/

# DeepLearningAssignment1

This project implements a deep neural network for classifying the Fashion MNIST dataset, complete with L2 regularization and several optimization techniques. All experiments are tracked using [Weights & Biases (wandb)](https://wandb.ai).

## Overview

The project is split into two main components:
- **Model & Training Logic (model.py):**  
  This module contains the neural network implementation, including:
  - **Neural Operations & Activation Functions:** ReLU, Sigmoid, and Tanh along with their gradients.
  - **DeepNeuralNet Class:** Handles multi-layer network creation, forward propagation (with softmax output), cost calculation (cross-entropy or mean squared error with L2 regularization), backpropagation, and parameter updates with several optimizers (SGD, Momentum, Nesterov, RMSProp, and Adam).
  - **Helper Functions:** For one-hot encoding, accuracy calculation, and plotting a confusion matrix.
  - **Experiment Logging:** Functions for logging sample images, dedicated experiment runs, and hyperparameter sweeps to wandb.
  
- **Experiment Orchestration (train.py):**  
  This script acts as the entry point. It:
  - Parses command-line arguments for wandb configuration (entity and project name).
  - Initiates logging of sample images from Fashion MNIST.
  - Runs dedicated experiments with preset configurations.
  - Launches a hyperparameter sweep to explore various configurations.
  
All training runs log metrics such as loss, training/validation accuracy, and test accuracy, as well as a confusion matrix for final evaluation.

## Project Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/SaiRajesh228/DeepLearningAssignment1.git
   cd DeepLearningAssignment1
   ```

2. **Install Dependencies:**

   Ensure you have Python (3.7 or above) installed, then install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Log in to Weights & Biases:**

   Before running the training script, log in to wandb using:

   ```bash
   wandb login
   ```

## Training and Evaluation

- **Training Process:**  
  The training pipeline involves:
  - **Data Preparation:** The Fashion MNIST dataset is loaded and normalized, with labels converted to one-hot encoding. A validation split is created from the training data.
  - **Forward Propagation:** The model computes outputs using the appropriate activation functions in hidden layers and softmax in the output layer.
  - **Cost Computation:** The loss is computed using cross-entropy or mean squared error, augmented by an L2 regularization term to help prevent overfitting.
  - **Backward Propagation:** Gradients are computed via backpropagation, including adjustments for L2 regularization.
  - **Parameter Update:** The model supports several optimizers (SGD, Momentum, Nesterov, RMSProp, Adam) for updating weights and biases.
  - **Logging:** After each epoch, training and validation accuracy, as well as the average loss, are logged to wandb.

- **Evaluation:**  
  After training, the model is evaluated on the test set:
  - **Final Accuracy:** The test set is passed through the model to compute the final classification accuracy.
  - **Confusion Matrix:** A confusion matrix is generated and logged to provide insight into which classes are being misclassified.

## Running the Project via Command-Line

The project is set up to be run from the command-line through `train.py`. This script requires two command-line arguments:
- **--wandb_entity:** Your wandb username or team name.  
  In this case, use your wandb entity (e.g., `karapa-rajesh`).
- **--wandb_project:** The wandb project name.  
  Here, use `DeepLearningAssignment1`.

### Example Command

Run the following command from the project root:

```bash
python train.py --wandb_entity karapa-rajesh --wandb_project DeepLearning
```

This command will:
1. Log sample images from the Fashion MNIST dataset.
2. Execute a series of dedicated experiments with pre-configured settings.
3. Initiate a hyperparameter sweep to optimize the model's performance.

All logs and metrics will be sent to your wandb dashboard under the project **DeepLearningAssignment1**.

## Additional Notes

- **Extensibility:**  
  The code is modular. You can adjust the network architecture, training parameters, and experiment configurations in the `model.py` file without changing the command-line interface in `train.py`.

- **Reproducibility:**  
  The clear separation between model logic and experiment orchestration makes it easy to reproduce the experiments and share them with collaborators.



