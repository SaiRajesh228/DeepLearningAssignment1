{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww12480\viewh9600\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs26 \cf0 import numpy as np\
import matplotlib.pyplot as plt\
import wandb\
from keras.datasets import fashion_mnist\
from sklearn.metrics import confusion_matrix\
import seaborn as sns\
\
# -----------------------\
# Neural operations\
def linear(z):\
    return np.maximum(0, z)\
\
def linear_grad(z):\
    return (z > 0) * 1.0\
\
def logistic(z):\
    return 1 / (1 + np.exp(-z))\
\
def logistic_grad(z):\
    s = logistic(z)\
    return s * (1 - s)\
\
def hyperbolic(z):\
    return np.tanh(z)\
\
def hyperbolic_grad(z):\
    return 1 - np.tanh(z)**2\
\
neural_ops = \{\
    "relu": (linear, linear_grad),\
    "sigmoid": (logistic, logistic_grad),\
    "tanh": (hyperbolic, hyperbolic_grad)\
\}\
\
# -----------------------\
# Deep Neural Network with L2 Regularization\
class DeepNeuralNet:\
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu", init_scheme="Xavier"):\
        self.layer_count = len(hidden_dims) + 1\
        self.activation = activation\
        self.weights = []\
        self.biases = []\
\
        dims = [input_dim] + hidden_dims + [output_dim]\
        for i in range(self.layer_count):\
            if init_scheme == "Xavier":\
                if activation in ["tanh", "sigmoid"]:\
                    scale = np.sqrt(2. / (dims[i] + dims[i+1]))\
                else:  # ReLU\
                    scale = np.sqrt(2. / dims[i])\
                W = np.random.randn(dims[i], dims[i+1]) * scale\
            else:  # Random initialization\
                W = np.random.randn(dims[i], dims[i+1]) * 0.01\
            b = np.zeros((1, dims[i+1]))\
            self.weights.append(W)\
            self.biases.append(b)\
\
    def predict(self, X):\
        activate, _ = neural_ops[self.activation]\
        self.z_records = []\
        self.a_records = [X]\
        A = X\
        for idx in range(self.layer_count):\
            Z = A.dot(self.weights[idx]) + self.biases[idx]\
            self.z_records.append(Z)\
            if idx == self.layer_count - 1:\
                # Use softmax for final layer\
                shifted = Z - np.max(Z, axis=1, keepdims=True)\
                exp = np.exp(shifted)\
                A = exp / np.sum(exp, axis=1, keepdims=True)\
            else:\
                A = activate(Z)\
            self.a_records.append(A)\
        return A\
\
    def calculate_cost(self, Y_hat, Y_real, cost_type="cross_entropy", weight_decay=0):\
        m = Y_real.shape[0]\
        if cost_type == "cross_entropy":\
            cost = -np.sum(Y_real * np.log(Y_hat + 1e-8)) / m\
        elif cost_type == "mean_squared_error":\
            cost = np.sum((Y_real - Y_hat)**2) / (2 * m)\
        if weight_decay > 0:\
            l2_penalty = sum(np.sum(w**2) for w in self.weights)\
            cost += (weight_decay / (2 * m)) * l2_penalty\
        return cost\
\
    def compute_gradients(self, X, Y, cost_type="cross_entropy", weight_decay=0):\
        m = X.shape[0]\
        grad_weights = [None] * self.layer_count\
        grad_biases = [None] * self.layer_count\
\
        final_act = self.a_records[-1]\
        if cost_type == "cross_entropy":\
            delta = final_act - Y\
        elif cost_type == "mean_squared_error":\
            delta = (final_act - Y)\
\
        for idx in reversed(range(self.layer_count)):\
            if idx == self.layer_count - 1:\
                dZ = delta\
            else:\
                _, grad_func = neural_ops[self.activation]\
                dZ = delta * grad_func(self.z_records[idx])\
            prev_act = self.a_records[idx]\
            grad_weights[idx] = prev_act.T.dot(dZ) / m\
            grad_biases[idx] = np.sum(dZ, axis=0, keepdims=True) / m\
            if weight_decay > 0:\
                grad_weights[idx] += (weight_decay / m) * self.weights[idx]\
            if idx > 0:\
                delta = dZ.dot(self.weights[idx].T)\
        return grad_weights, grad_biases\
\
    def adjust_params(self, grad_w, grad_b, optim, settings, states):\
        lr = settings.learning_rate\
\
        if optim == "sgd":\
            for i in range(self.layer_count):\
                self.weights[i] -= lr * grad_w[i]\
                self.biases[i] -= lr * grad_b[i]\
\
        elif optim == "momentum":\
            momentum_val = 0.9\
            if "momentum" not in states:\
                states["momentum"] = \{\
                    "v_w": [np.zeros_like(w) for w in self.weights],\
                    "v_b": [np.zeros_like(b) for b in self.biases]\
                \}\
            for i in range(self.layer_count):\
                states["momentum"]["v_w"][i] = momentum_val * states["momentum"]["v_w"][i] + grad_w[i]\
                self.weights[i] -= lr * states["momentum"]["v_w"][i]\
                states["momentum"]["v_b"][i] = momentum_val * states["momentum"]["v_b"][i] + grad_b[i]\
                self.biases[i] -= lr * states["momentum"]["v_b"][i]\
\
        elif optim == "nesterov":\
            momentum = 0.9\
            if "nesterov" not in states:\
                states["nesterov"] = \{\
                    "v_w": [np.zeros_like(w) for w in self.weights],\
                    "v_b": [np.zeros_like(b) for b in self.biases]\
                \}\
            for i in range(self.layer_count):\
                states["nesterov"]["v_w"][i] = momentum * states["nesterov"]["v_w"][i] + grad_w[i]\
                self.weights[i] -= lr * (momentum * states["nesterov"]["v_w"][i] + grad_w[i])\
                states["nesterov"]["v_b"][i] = momentum * states["nesterov"]["v_b"][i] + grad_b[i]\
                self.biases[i] -= lr * (momentum * states["nesterov"]["v_b"][i] + grad_b[i])\
\
        elif optim == "rmsprop":\
            gamma = 0.9\
            eps = 1e-8\
            if "rmsprop" not in states:\
                states["rmsprop"] = \{\
                    "cache_w": [np.zeros_like(w) for w in self.weights],\
                    "cache_b": [np.zeros_like(b) for b in self.biases]\
                \}\
            for i in range(self.layer_count):\
                states["rmsprop"]["cache_w"][i] = gamma * states["rmsprop"]["cache_w"][i] + (1 - gamma) * (grad_w[i]**2)\
                self.weights[i] -= lr * grad_w[i] / (np.sqrt(states["rmsprop"]["cache_w"][i]) + eps)\
                states["rmsprop"]["cache_b"][i] = gamma * states["rmsprop"]["cache_b"][i] + (1 - gamma) * (grad_b[i]**2)\
                self.biases[i] -= lr * grad_b[i] / (np.sqrt(states["rmsprop"]["cache_b"][i]) + eps)\
\
        elif optim == "adam":\
            beta1 = 0.9\
            beta2 = 0.999\
            eps = 1e-8\
            if "adam" not in states:\
                states["adam"] = \{\
                    "m_w": [np.zeros_like(w) for w in self.weights],\
                    "v_w": [np.zeros_like(w) for w in self.weights],\
                    "m_b": [np.zeros_like(b) for b in self.biases],\
                    "v_b": [np.zeros_like(b) for b in self.biases],\
                    "step": 0\
                \}\
            states["adam"]["step"] += 1\
            t = states["adam"]["step"]\
            for i in range(self.layer_count):\
                states["adam"]["m_w"][i] = beta1 * states["adam"]["m_w"][i] + (1 - beta1) * grad_w[i]\
                states["adam"]["v_w"][i] = beta2 * states["adam"]["v_w"][i] + (1 - beta2) * (grad_w[i]**2)\
                m_w_adj = states["adam"]["m_w"][i] / (1 - beta1**t)\
                v_w_adj = states["adam"]["v_w"][i] / (1 - beta2**t)\
                self.weights[i] -= lr * m_w_adj / (np.sqrt(v_w_adj) + eps)\
\
                states["adam"]["m_b"][i] = beta1 * states["adam"]["m_b"][i] + (1 - beta1) * grad_b[i]\
                states["adam"]["v_b"][i] = beta2 * states["adam"]["v_b"][i] + (1 - beta2) * (grad_b[i]**2)\
                m_b_adj = states["adam"]["m_b"][i] / (1 - beta1**t)\
                v_b_adj = states["adam"]["v_b"][i] / (1 - beta2**t)\
                self.biases[i] -= lr * m_b_adj / (np.sqrt(v_b_adj) + eps)\
        return states\
\
# -----------------------\
# Helper utilities\
def encode_labels(y, num_labels):\
    encoded = np.zeros((len(y), num_labels))\
    encoded[np.arange(len(y)), y] = 1\
    return encoded\
\
def get_accuracy(Y_est, Y_actual):\
    preds = np.argmax(Y_est, axis=1)\
    truths = np.argmax(Y_actual, axis=1)\
    return np.mean(preds == truths)\
\
def log_confusion_matrix(Y_est, y_real, classes):\
    cm = confusion_matrix(y_real, np.argmax(Y_est, axis=1))\
    plt.figure(figsize=(9,7))\
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)\
    plt.xlabel("Predicted")\
    plt.ylabel("True")\
    plt.title("Confusion Matrix")\
    wandb.log(\{"Confusion Matrix": wandb.Image(plt)\})\
    plt.close()\
\
# -----------------------\
# Question 1: Log sample images\
def log_q1_samples():\
    (train_X, train_y), _ = fashion_mnist.load_data()\
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',\
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']\
    plt.figure(figsize=(10,5))\
    for i in range(10):\
        idx = np.where(train_y == i)[0][0]\
        plt.subplot(2,5,i+1)\
        plt.imshow(train_X[idx], cmap='gray')\
        plt.title(class_names[i])\
        plt.axis('off')\
    plt.tight_layout()\
    wandb.log(\{"Question 1 Samples": wandb.Image(plt)\})\
    plt.close()\
\
# -----------------------\
# Dedicated Question 2 Logging\
def log_q2_experiments():\
    q2_configs = [\
        \{\
            "name": "Q2a_Sigmoid_NoReg",\
            "config": \{\
                "activation_func": "sigmoid",\
                "weight_decay": 0,\
                "hiddenlayers": 3,\
                "hiddennodes": 128,\
                "num_epochs": 10,\
                "learning_rate": 0.001,\
                "batch_size": 64,\
                "opt": "adam",\
                "loss": "cross_entropy",\
                "initializer": "Xavier"\
            \}\
        \},\
        \{\
            "name": "Q2b_Tanh_L2",\
            "config": \{\
                "activation_func": "tanh",\
                "weight_decay": 0.001,\
                "hiddenlayers": 3,\
                "hiddennodes": 128,\
                "num_epochs": 10,\
                "learning_rate": 0.001,\
                "batch_size": 64,\
                "opt": "adam",\
                "loss": "cross_entropy",\
                "initializer": "Xavier"\
            \}\
        \},\
        \{\
            "name": "Q2c_ReLU_L2",\
            "config": \{\
                "activation_func": "relu",\
                "weight_decay": 0.0001,\
                "hiddenlayers": 3,\
                "hiddennodes": 128,\
                "num_epochs": 10,\
                "learning_rate": 0.001,\
                "batch_size": 64,\
                "opt": "adam",\
                "loss": "cross_entropy",\
                "initializer": "Xavier"\
            \}\
        \}\
    ]\
\
    for exp in q2_configs:\
        with wandb.init(name=exp["name"], config=exp["config"], tags=["Question2"]):\
            execute_training()\
        wandb.finish()\
\
# -----------------------\
# Enhanced Training Procedure with L2 Regularization\
def execute_training():\
    wandb.init()\
    cfg = wandb.config\
\
    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()\
    train_X = train_X.reshape(train_X.shape[0], -1) / 255.0\
    test_X = test_X.reshape(test_X.shape[0], -1) / 255.0\
    num_classes = 10\
    train_y_oh = encode_labels(train_y, num_classes)\
    test_y_oh = encode_labels(test_y, num_classes)\
\
    # Validation split\
    val_split = int(0.9 * train_X.shape[0])\
    val_X, val_y_oh = train_X[val_split:], train_y_oh[val_split:]\
    train_X, train_y_oh = train_X[:val_split], train_y_oh[:val_split]\
\
    input_dim = train_X.shape[1]\
    hidden_arch = [cfg.hiddennodes] * cfg.hiddenlayers\
    model = DeepNeuralNet(input_dim, hidden_arch, num_classes,\
                          activation=cfg.activation_func,\
                          init_scheme=cfg.initializer)\
\
    optimizer_states = \{\}\
    grad_clip_value = 1.0\
\
    for epoch in range(cfg.num_epochs):\
        shuffle_idx = np.random.permutation(train_X.shape[0])\
        train_X = train_X[shuffle_idx]\
        train_y_oh = train_y_oh[shuffle_idx]\
\
        batches = train_X.shape[0] // cfg.batch_size\
        epoch_loss = 0.0\
\
        for batch in range(batches):\
            start = batch * cfg.batch_size\
            end = start + cfg.batch_size\
            X_batch = train_X[start:end]\
            y_batch = train_y_oh[start:end]\
\
            outputs = model.predict(X_batch)\
            loss = model.calculate_cost(outputs, y_batch,\
                                        cost_type=cfg.loss,\
                                        weight_decay=cfg.weight_decay)\
            epoch_loss += loss\
\
            grad_w, grad_b = model.compute_gradients(X_batch, y_batch,\
                                                     cost_type=cfg.loss,\
                                                     weight_decay=cfg.weight_decay)\
            for i in range(len(grad_w)):\
                grad_w[i] = np.clip(grad_w[i], -grad_clip_value, grad_clip_value)\
                grad_b[i] = np.clip(grad_b[i], -grad_clip_value, grad_clip_value)\
\
            optimizer_states = model.adjust_params(grad_w, grad_b, cfg.opt, cfg, optimizer_states)\
\
        avg_loss = epoch_loss / batches\
        train_outputs = model.predict(train_X)\
        train_acc = get_accuracy(train_outputs, train_y_oh)\
        val_outputs = model.predict(val_X)\
        val_acc = get_accuracy(val_outputs, val_y_oh)\
\
        wandb.log(\{\
            "epoch": epoch+1,\
            "loss": avg_loss,\
            "train_accuracy": train_acc,\
            "val_accuracy": val_acc\
        \})\
\
    test_outputs = model.predict(test_X)\
    final_acc = get_accuracy(test_outputs, test_y_oh)\
    wandb.log(\{"test_accuracy": final_acc\})\
    log_confusion_matrix(test_outputs, test_y, [str(i) for i in range(num_classes)])\
    wandb.finish()\
\
# -----------------------\
# Question 2 Sweep Configuration\
sweep_config = \{\
    'name': "question2-sweep",\
    'method': 'bayes',\
    'metric': \{'name': 'val_accuracy', 'goal': 'maximize'\},\
    'parameters': \{\
        'hiddenlayers': \{'values': [3, 4, 5]\},\
        'num_epochs': \{'values': [10, 15]\},\
        'hiddennodes': \{'values': [128, 256]\},\
        'learning_rate': \{'values': [1e-3, 5e-4]\},\
        'initializer': \{'values': ["Xavier", "random"]\},\
        'batch_size': \{'values': [64, 128]\},\
        'opt': \{'values': ["adam", "nesterov", "rmsprop"]\},\
        'activation_func': \{'values': ["relu", "tanh", "sigmoid"]\},\
        'loss': \{'values': ["cross_entropy", "mean_squared_error"]\},\
        'weight_decay': \{'values': [0, 0.0001, 0.001]\}\
    \}\
\}\
}