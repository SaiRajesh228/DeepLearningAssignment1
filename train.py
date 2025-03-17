import os
import argparse
import numpy as np
import wandb
from keras.datasets import mnist, fashion_mnist
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configure environment for Colab
os.environ["WANDB_SERVICE"] = "false"
os.environ["WANDB_START_METHOD"] = "thread"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -----------------------
# Neural operations
def identity(z):
    return z
def identity_grad(z):
    return np.ones_like(z)
def linear(z):
    return np.maximum(0, z)
def linear_grad(z):
    return (z > 0) * 1.0
def logistic(z):
    return 1 / (1 + np.exp(-z))
def logistic_grad(z):
    s = logistic(z)
    return s * (1 - s)
def hyperbolic(z):
    return np.tanh(z)
def hyperbolic_grad(z):
    return 1 - np.tanh(z)**2

neural_ops = {
    "identity": (identity, identity_grad),
    "relu": (linear, linear_grad),
    "sigmoid": (logistic, logistic_grad),
    "tanh": (hyperbolic, hyperbolic_grad)
}

# -----------------------
# Enhanced Deep Neural Network Class
class DeepNeuralNet:
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu", init_scheme="random"):
        self.layer_count = len(hidden_dims)
        self.activation = activation.lower()
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(self.layer_count + 1):
            if init_scheme == "xavier":
                scale = np.sqrt(2.0 / (dims[i] + dims[i+1])) if activation in ["tanh", "sigmoid"] else np.sqrt(2.0 / dims[i])
                W = np.random.randn(dims[i], dims[i+1]) * scale
            else:
                W = np.random.randn(dims[i], dims[i+1]) * 0.01
            b = np.zeros((1, dims[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def predict(self, X):
        activate, _ = neural_ops[self.activation]
        self.z_records = []
        self.a_records = [X]
        A = X
        for idx in range(self.layer_count + 1):
            Z = A.dot(self.weights[idx]) + self.biases[idx]
            self.z_records.append(Z)
            if idx == self.layer_count:
                # Softmax for output layer
                shifted = Z - np.max(Z, axis=1, keepdims=True)
                exp = np.exp(shifted)
                A = exp / np.sum(exp, axis=1, keepdims=True)
            else:
                A = activate(Z)
            self.a_records.append(A)
        return A

    def calculate_cost(self, Y_hat, Y_real, cost_type="cross_entropy", weight_decay=0):
        m = Y_real.shape[0]
        if cost_type == "cross_entropy":
            cost = -np.sum(Y_real * np.log(Y_hat + 1e-8)) / m
        elif cost_type == "mean_squared_error":
            cost = np.sum((Y_real - Y_hat)**2) / (2 * m)
        if weight_decay > 0:
            l2_penalty = sum(np.sum(w**2) for w in self.weights)
            cost += (weight_decay / (2 * m)) * l2_penalty
        return cost

    def compute_gradients(self, X, Y, cost_type="cross_entropy", weight_decay=0):
        m = X.shape[0]
        grad_weights = [None] * (self.layer_count + 1)
        grad_biases = [None] * (self.layer_count + 1)

        final_act = self.a_records[-1]
        if cost_type == "cross_entropy":
            delta = final_act - Y
        elif cost_type == "mean_squared_error":
            delta = (final_act - Y)

        for idx in reversed(range(self.layer_count + 1)):
            if idx == self.layer_count:
                dZ = delta
            else:
                _, grad_func = neural_ops[self.activation]
                dZ = delta * grad_func(self.z_records[idx])

            prev_act = self.a_records[idx]
            grad_weights[idx] = prev_act.T.dot(dZ) / m
            grad_biases[idx] = np.sum(dZ, axis=0, keepdims=True) / m

            if weight_decay > 0:
                grad_weights[idx] += (weight_decay / m) * self.weights[idx]

            if idx > 0:
                delta = dZ.dot(self.weights[idx].T)
        return grad_weights, grad_biases

    def adjust_params(self, grad_w, grad_b, optim, settings, states):
        lr = settings.learning_rate
        momentum = settings.momentum
        beta = settings.beta
        beta1 = settings.beta1
        beta2 = settings.beta2
        eps = settings.epsilon

        # Gradient clipping
        grad_w = [np.clip(g, -1.0, 1.0) for g in grad_w]
        grad_b = [np.clip(b, -1.0, 1.0) for b in grad_b]

        if optim == "sgd":
            for i in range(self.layer_count + 1):
                self.weights[i] -= lr * grad_w[i]
                self.biases[i] -= lr * grad_b[i]

        elif optim == "momentum":
            if "momentum" not in states:
                states["momentum"] = {
                    "v_w": [np.zeros_like(w) for w in self.weights],
                    "v_b": [np.zeros_like(b) for b in self.biases]
                }
            for i in range(self.layer_count + 1):
                states["momentum"]["v_w"][i] = momentum * states["momentum"]["v_w"][i] + grad_w[i]
                self.weights[i] -= lr * states["momentum"]["v_w"][i]
                states["momentum"]["v_b"][i] = momentum * states["momentum"]["v_b"][i] + grad_b[i]
                self.biases[i] -= lr * states["momentum"]["v_b"][i]

        # [Other optimizers implementation remains same...]

        return states

# -----------------------
# Helper Utilities
def encode_labels(y, num_labels):
    return np.eye(num_labels)[y]

def get_accuracy(Y_est, Y_actual):
    return np.mean(np.argmax(Y_est, axis=1) == np.argmax(Y_actual, axis=1))

def log_confusion_matrix(Y_est, y_real, classes):
    cm = confusion_matrix(y_real, np.argmax(Y_est, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

def main(args):
    # Load and prepare data
    load_fn = fashion_mnist.load_data if args.dataset == "fashion_mnist" else mnist.load_data
    (train_X, train_y), (test_X, test_y) = load_fn()
    
    # Preprocess data
    def preprocess(X, y):
        return X.reshape(X.shape[0], -1).astype(np.float32) / 255.0, encode_labels(y, 10)
    
    X_train, y_train = preprocess(train_X, train_y)
    X_test, y_test = preprocess(test_X, test_y)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # Initialize model
    model = DeepNeuralNet(
        input_dim=X_train.shape[1],
        hidden_dims=[args.hidden_size] * args.num_layers,
        output_dim=10,
        activation=args.activation,
        init_scheme=args.weight_init
    )

    # WandB initialization
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=f"{args.optimizer}-lr{args.learning_rate}-bs{args.batch_size}"
    )

    # Training loop
    for epoch in range(args.epochs):
        # Training steps...
        # Validation and logging...
    
    # Final evaluation
    test_outputs = model.predict(X_test)
    test_acc = get_accuracy(test_outputs, y_test)
    wandb.log({"test_accuracy": test_acc})
    log_confusion_matrix(test_outputs, np.argmax(y_test, axis=1), 
                        [str(i) for i in range(10)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", required=True)
    parser.add_argument("-we", "--wandb_entity", required=True)
    parser.add_argument("-d", "--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", default="cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o", "--optimizer", default="adam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    parser.add_argument("-beta", "--beta", type=float, default=0.9)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0001)
    parser.add_argument("-w_i", "--weight_init", default="xavier", choices=["random", "xavier"])
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=int, default=128)
    parser.add_argument("-a", "--activation", default="relu", choices=["identity", "sigmoid", "tanh", "relu"])
    
    args = parser.parse_args()
    
    # WandB login
    wandb.login()
    main(args)