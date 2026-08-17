"""
Federated Learning Core: Simple Neural Network + FL Protocol.

Implements a lightweight MLP for medical image classification,
designed to run without PyTorch (pure numpy + scipy).
"""

import numpy as np
from typing import List, Tuple, Dict


# ============================================================
# Activation Functions
# ============================================================
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float64)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ============================================================
# Simple MLP (Multi-Layer Perceptron)
# ============================================================
class SimpleMLP:
    """
    Two-hidden-layer MLP for classification.
    Default architecture: input_dim -> 128 -> 64 -> n_classes.
    Use hidden=(32, 16) for full-vector HE on tabular medical data.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        seed: int = 42,
        hidden: Tuple[int, int] = (128, 64),
    ):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.h1, self.h2 = int(hidden[0]), int(hidden[1])
        self.rng = np.random.default_rng(seed)

        self.W1 = self.rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, self.h1))
        self.b1 = np.zeros(self.h1)
        self.W2 = self.rng.normal(0, np.sqrt(2.0 / self.h1), (self.h1, self.h2))
        self.b2 = np.zeros(self.h2)
        self.W3 = self.rng.normal(0, np.sqrt(2.0 / self.h2), (self.h2, n_classes))
        self.b3 = np.zeros(n_classes)

    def get_weights(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        return np.concatenate([
            self.W1.ravel(), self.b1,
            self.W2.ravel(), self.b2,
            self.W3.ravel(), self.b3,
        ])

    def set_weights(self, flat: np.ndarray):
        """Set parameters from a flat vector."""
        idx = 0
        size = self.input_dim * self.h1
        self.W1 = flat[idx:idx + size].reshape(self.input_dim, self.h1)
        idx += size
        self.b1 = flat[idx:idx + self.h1]
        idx += self.h1
        size = self.h1 * self.h2
        self.W2 = flat[idx:idx + size].reshape(self.h1, self.h2)
        idx += size
        self.b2 = flat[idx:idx + self.h2]
        idx += self.h2
        size = self.h2 * self.n_classes
        self.W3 = flat[idx:idx + size].reshape(self.h2, self.n_classes)
        idx += size
        self.b3 = flat[idx:idx + self.n_classes]

    def n_params(self) -> int:
        return len(self.get_weights())

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass, returns predictions and cache for backprop."""
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3
        a3 = softmax(z3)

        cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3, 'a3': a3}
        return a3, cache

    def backward(self, y_onehot: np.ndarray, cache: Dict) -> Dict:
        """Backward pass, returns gradients as a dict."""
        m = y_onehot.shape[0]

        dz3 = (cache['a3'] - y_onehot) / m
        dW3 = cache['a2'].T @ dz3
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_deriv(cache['z2'])
        dW2 = cache['a1'].T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_deriv(cache['z1'])
        dW1 = cache['X'].T @ dz1
        db1 = dz1.sum(axis=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

    def get_gradient_vector(self, grads: Dict) -> np.ndarray:
        """Flatten gradients into a single vector."""
        return np.concatenate([
            grads['dW1'].ravel(), grads['db1'],
            grads['dW2'].ravel(), grads['db2'],
            grads['dW3'].ravel(), grads['db3'],
        ])

    def compute_loss(self, y_pred: np.ndarray, y_onehot: np.ndarray) -> float:
        """Cross-entropy loss."""
        eps = 1e-12
        return -np.mean(np.sum(y_onehot * np.log(y_pred + eps), axis=1))

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01) -> Tuple[np.ndarray, float]:
        """Single training step. Returns gradient vector and loss."""
        y_onehot = np.zeros((len(y), self.n_classes))
        y_onehot[np.arange(len(y)), y.astype(int)] = 1

        y_pred, cache = self.forward(X)
        loss = self.compute_loss(y_pred, y_onehot)

        grads = self.backward(y_onehot, cache)
        grad_vec = self.get_gradient_vector(grads)

        return grad_vec, loss

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate accuracy and loss."""
        y_onehot = np.zeros((len(y), self.n_classes))
        y_onehot[np.arange(len(y)), y.astype(int)] = 1

        y_pred, _ = self.forward(X)
        loss = self.compute_loss(y_pred, y_onehot)
        acc = np.mean(np.argmax(y_pred, axis=1) == y)
        return acc, loss


# ============================================================
# Synthetic Medical Data Generator
# ============================================================
def generate_synthetic_medical_data(
    n_samples: int = 200,
    n_features: int = 784,
    n_classes: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic stand-in for medical imaging features."""
    rng = np.random.default_rng(seed)
    X_all, y_all = [], []
    samples_per_class = n_samples // n_classes
    for cls in range(n_classes):
        mean = rng.normal(0, 0.5, n_features)
        mean[: n_features // n_classes * (cls + 1)] += 0.3 * (cls + 1)
        X_cls = rng.normal(mean, 0.3, size=(samples_per_class, n_features))
        for i in range(samples_per_class):
            for j in range(1, n_features):
                X_cls[i, j] += 0.2 * X_cls[i, j - 1]
        X_all.append(X_cls)
        y_all.append(np.full(samples_per_class, cls))
    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ============================================================
# Real medical data loaders
# ============================================================
def load_breast_cancer_medical(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    UCI Breast Cancer Wisconsin (diagnostic) via scikit-learn.
    Real clinical features (30), binary labels — public medical dataset.
    """
    try:
        from sklearn.datasets import load_breast_cancer
    except ImportError as e:
        raise ImportError("scikit-learn required for real medical data: pip install scikit-learn") from e

    bundle = load_breast_cancer()
    X = bundle.data.astype(np.float64)
    y = bundle.target.astype(np.int64)
    # standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    meta = {
        "name": "UCI Breast Cancer Wisconsin",
        "n_features": X.shape[1],
        "n_classes": int(len(np.unique(y))),
        "n_samples": int(len(X)),
        "source": "sklearn.datasets.load_breast_cancer",
    }
    return X[perm], y[perm], meta


def load_medical_dataset(
    name: str = "breast_cancer", seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load a public medical dataset. Default: breast_cancer (always offline)."""
    name = name.lower()
    if name in ("breast_cancer", "breast", "uci"):
        return load_breast_cancer_medical(seed)
    if name in ("synthetic", "synth"):
        X, y = generate_synthetic_medical_data(seed=seed)
        meta = {"name": "synthetic", "n_features": X.shape[1], "n_classes": 4, "n_samples": len(X)}
        return X, y, meta
    # Optional MedMNIST if installed
    if name.startswith("medmnist") or name in ("pneumoniamnist", "pathmnist", "bloodmnist"):
        try:
            import medmnist
            from medmnist import INFO
        except ImportError as e:
            raise ImportError(
                "medmnist not installed. pip install medmnist  OR use name='breast_cancer'"
            ) from e
        key = name.replace("medmnist_", "").replace("medmnist", "pneumoniamnist")
        if key not in INFO:
            key = "pneumoniamnist"
        DataClass = getattr(medmnist, INFO[key]["python_class"])
        train = DataClass(split="train", download=True)
        # flatten images
        imgs = train.imgs
        if imgs.ndim == 4:
            X = imgs.reshape(len(imgs), -1).astype(np.float64) / 255.0
        else:
            X = imgs.reshape(len(imgs), -1).astype(np.float64) / 255.0
        y = train.labels.ravel().astype(np.int64)
        # subsample for FL prototype speed
        rng = np.random.default_rng(seed)
        if len(X) > 2000:
            idx = rng.choice(len(X), size=2000, replace=False)
            X, y = X[idx], y[idx]
        X = (X - X.mean()) / (X.std() + 1e-8)
        meta = {
            "name": key,
            "n_features": X.shape[1],
            "n_classes": int(len(np.unique(y))),
            "n_samples": int(len(X)),
            "source": "medmnist",
            "projected": False,
            "full_resolution": True,
        }
        return X, y, meta
    raise ValueError(f"Unknown medical dataset: {name}")



def partition_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition data among clients in a non-IID fashion using Dirichlet distribution."""
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(y))

    class_indices = [np.where(y == c)[0] for c in range(n_classes)]
    client_data = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        proportions = rng.dirichlet(np.ones(n_clients) * alpha)
        proportions = (proportions * len(class_indices[c])).astype(int)
        proportions[-1] = len(class_indices[c]) - proportions[:-1].sum()

        idx = rng.permutation(class_indices[c])
        start = 0
        for k in range(n_clients):
            end = start + proportions[k]
            client_data[k].extend(idx[start:end].tolist())
            start = end

    partitions = []
    for k in range(n_clients):
        indices = np.array(client_data[k])
        if len(indices) > 0:
            rng.shuffle(indices)
            partitions.append((X[indices], y[indices]))
        else:
            fallback = rng.choice(len(X), size=10, replace=False)
            partitions.append((X[fallback], y[fallback]))

    return partitions
