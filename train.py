import argparse
import numpy as np
import wandb
from keras.datasets import mnist, fashion_mnist
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
        self.activation = activation.lower() if activation != "ReLU" else "relu"
        self.weights = []
        self.biases = []

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(self.layer_count + 1):
            if init_scheme == "xavier":
                if self.activation in ["tanh", "sigmoid"]:
                    scale = np.sqrt(2. / (dims[i] + dims[i+1]))
                else:
                    scale = np.sqrt(2. / dims[i])
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

        elif optim == "nag":
            if "nag" not in states:
                states["nag"] = {
                    "v_w": [np.zeros_like(w) for w in self.weights],
                    "v_b": [np.zeros_like(b) for b in self.biases]
                }
            for i in range(self.layer_count + 1):
                v_w_prev = states["nag"]["v_w"][i].copy()
                states["nag"]["v_w"][i] = momentum * states["nag"]["v_w"][i] + grad_w[i]
                self.weights[i] -= lr * (momentum * states["nag"]["v_w"][i] + grad_w[i])
                v_b_prev = states["nag"]["v_b"][i].copy()
                states["nag"]["v_b"][i] = momentum * states["nag"]["v_b"][i] + grad_b[i]
                self.biases[i] -= lr * (momentum * states["nag"]["v_b"][i] + grad_b[i])

        elif optim == "rmsprop":
            if "rmsprop" not in states:
                states["rmsprop"] = {
                    "cache_w": [np.zeros_like(w) for w in self.weights],
                    "cache_b": [np.zeros_like(b) for b in self.biases]
                }
            for i in range(self.layer_count + 1):
                states["rmsprop"]["cache_w"][i] = beta * states["rmsprop"]["cache_w"][i] + (1 - beta) * (grad_w[i]**2)
                self.weights[i] -= lr * grad_w[i] / (np.sqrt(states["rmsprop"]["cache_w"][i]) + eps)
                states["rmsprop"]["cache_b"][i] = beta * states["rmsprop"]["cache_b"][i] + (1 - beta) * (grad_b[i]**2)
                self.biases[i] -= lr * grad_b[i] / (np.sqrt(states["rmsprop"]["cache_b"][i]) + eps)

        elif optim == "adam":
            beta1 = settings.beta1
            beta2 = settings.beta2
            if "adam" not in states:
                states["adam"] = {
                    "m_w": [np.zeros_like(w) for w in self.weights],
                    "v_w": [np.zeros_like(w) for w in self.weights],
                    "m_b": [np.zeros_like(b) for b in self.biases],
                    "v_b": [np.zeros_like(b) for b in self.biases],
                    "step": 0
                }
            states["adam"]["step"] += 1
            t = states["adam"]["step"]
            for i in range(self.layer_count + 1):
                states["adam"]["m_w"][i] = beta1 * states["adam"]["m_w"][i] + (1 - beta1) * grad_w[i]
                states["adam"]["v_w"][i] = beta2 * states["adam"]["v_w"][i] + (1 - beta2) * (grad_w[i]**2)
                m_w_adj = states["adam"]["m_w"][i] / (1 - beta1**t)
                v_w_adj = states["adam"]["v_w"][i] / (1 - beta2**t)
                self.weights[i] -= lr * m_w_adj / (np.sqrt(v_w_adj) + eps)

                states["adam"]["m_b"][i] = beta1 * states["adam"]["m_b"][i] + (1 - beta1) * grad_b[i]
                states["adam"]["v_b"][i] = beta2 * states["adam"]["v_b"][i] + (1 - beta2) * (grad_b[i]**2)
                m_b_adj = states["adam"]["m_b"][i] / (1 - beta1**t)
                v_b_adj = states["adam"]["v_b"][i] / (1 - beta2**t)
                self.biases[i] -= lr * m_b_adj / (np.sqrt(v_b_adj) + eps)

        elif optim == "nadam":
            beta1 = settings.beta1
            beta2 = settings.beta2
            if "nadam" not in states:
                states["nadam"] = {
                    "m_w": [np.zeros_like(w) for w in self.weights],
                    "v_w": [np.zeros_like(w) for w in self.weights],
                    "m_b": [np.zeros_like(b) for b in self.biases],
                    "v_b": [np.zeros_like(b) for b in self.biases],
                    "step": 0
                }
            states["nadam"]["step"] += 1
            t = states["nadam"]["step"]
            for i in range(self.layer_count + 1):
                states["nadam"]["m_w"][i] = beta1 * states["nadam"]["m_w"][i] + (1 - beta1) * grad_w[i]
                states["nadam"]["v_w"][i] = beta2 * states["nadam"]["v_w"][i] + (1 - beta2) * (grad_w[i]**2)
                m_w_adj = (beta1 * states["nadam"]["m_w"][i] / (1 - beta1**t)) + ((1 - beta1) * grad_w[i] / (1 - beta1**t))
                v_w_adj = states["nadam"]["v_w"][i] / (1 - beta2**t)
                self.weights[i] -= lr * m_w_adj / (np.sqrt(v_w_adj) + eps)

                states["nadam"]["m_b"][i] = beta1 * states["nadam"]["m_b"][i] + (1 - beta1) * grad_b[i]
                states["nadam"]["v_b"][i] = beta2 * states["nadam"]["v_b"][i] + (1 - beta2) * (grad_b[i]**2)
                m_b_adj = (beta1 * states["nadam"]["m_b"][i] / (1 - beta1**t)) + ((1 - beta1) * grad_b[i] / (1 - beta1**t))
                v_b_adj = states["nadam"]["v_b"][i] / (1 - beta2**t)
                self.biases[i] -= lr * m_b_adj / (np.sqrt(v_b_adj) + eps)

        return states

# -----------------------
# Helper Utilities
def encode_labels(y, num_labels):
    encoded = np.zeros((len(y), num_labels))
    encoded[np.arange(len(y)), y] = 1
    return encoded

def get_accuracy(Y_est, Y_actual):
    preds = np.argmax(Y_est, axis=1)
    truths = np.argmax(Y_actual, axis=1)
    return np.mean(preds == truths)

def log_confusion_matrix(Y_est, y_real, classes):
    cm = confusion_matrix(y_real, np.argmax(Y_est, axis=1))
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    wandb.log({"Confusion Matrix": wandb.Image(plt)})
    plt.close()

# -----------------------
# Training Procedure
def train(config=None):
    with wandb.init(config=config) as run:
        cfg = run.config
        np.random.seed(42)

        # Load dataset
        if cfg.dataset == "mnist":
            (train_X, train_y), (test_X, test_y) = mnist.load_data()
        else:
            (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

        # Preprocess
        train_X = train_X.reshape(train_X.shape[0], -1) / 255.0
        test_X = test_X.reshape(test_X.shape[0], -1) / 255.0
        num_classes = 10
        train_y_oh = encode_labels(train_y, num_classes)
        test_y_oh = encode_labels(test_y, num_classes)

        # Split validation
        val_split = int(0.9 * train_X.shape[0])
        val_X, val_y_oh = train_X[val_split:], train_y_oh[val_split:]
        train_X, train_y_oh = train_X[:val_split], train_y_oh[:val_split]

        # Model setup
        input_dim = train_X.shape[1]
        hidden_arch = [cfg.hidden_size] * cfg.num_layers
        activation = cfg.activation.lower() if cfg.activation != "ReLU" else "relu"
        model = DeepNeuralNet(input_dim, hidden_arch, num_classes,
                              activation=activation,
                              init_scheme=cfg.weight_init)

        # Training loop
        optimizer_states = {}
        for epoch in range(cfg.epochs):
            shuffle_idx = np.random.permutation(train_X.shape[0])
            train_X = train_X[shuffle_idx]
            train_y_oh = train_y_oh[shuffle_idx]

            batches = train_X.shape[0] // cfg.batch_size
            epoch_loss = 0.0

            for batch in range(batches):
                start = batch * cfg.batch_size
                end = start + cfg.batch_size
                X_batch = train_X[start:end]
                y_batch = train_y_oh[start:end]

                # Forward pass
                outputs = model.predict(X_batch)
                loss = model.calculate_cost(outputs, y_batch, cfg.loss, cfg.weight_decay)
                epoch_loss += loss

                # Backward pass
                grad_w, grad_b = model.compute_gradients(X_batch, y_batch, cfg.loss, cfg.weight_decay)

                # Gradient clipping
                for i in range(len(grad_w)):
                    grad_w[i] = np.clip(grad_w[i], -1.0, 1.0)
                    grad_b[i] = np.clip(grad_b[i], -1.0, 1.0)

                # Update parameters
                optimizer_states = model.adjust_params(grad_w, grad_b, cfg.optimizer, cfg, optimizer_states)

            # Log metrics
            avg_loss = epoch_loss / batches
            train_outputs = model.predict(train_X)
            train_acc = get_accuracy(train_outputs, train_y_oh)
            val_outputs = model.predict(val_X)
            val_acc = get_accuracy(val_outputs, val_y_oh)

            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc
            })

        # Test evaluation
        test_outputs = model.predict(test_X)
        test_acc = get_accuracy(test_outputs, test_y_oh)
        wandb.log({"test_accuracy": test_acc})
        log_confusion_matrix(test_outputs, test_y, [str(i) for i in range(num_classes)])

# -----------------------
# Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o", "--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-w_i", "--weight_init", type=str, default="random", choices=["random", "xavier"])
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-a", "--activation", type=str, default="sigmoid", choices=["identity", "sigmoid", "tanh", "ReLU"])
    args = parser.parse_args()

    # Map activation to lowercase except ReLU
    if args.activation == "ReLU":
        args.activation = "relu"
    else:
        args.activation = args.activation.lower()

    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args
    )
    train()
    wandb.finish()