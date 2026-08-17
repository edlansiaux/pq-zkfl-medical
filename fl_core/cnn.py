"""
Small ConvNet for 28x28 medical images (no compact MLP head / no projection).

Architecture (NumPy):
  Conv(1→8,3x3,pad1) → ReLU → MaxPool2
  Conv(8→16,3x3,pad1) → ReLU → MaxPool2
  Flatten (16×7×7=784) → FC(784→64) → ReLU → FC(64→C)

Same flat weight / train_step / evaluate API as SimpleMLP for HE+ZKP.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x: (N,C,H,W), w: (Cout,Cin,k,k), pad=1, stride=1 → same H,W."""
    n, c_in, h, w_ = x.shape
    c_out, _, k, _ = w.shape
    assert k == 3
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((n, c_out, h, w_), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            # (N,Cin,H,W) * (Cout,Cin) via tensordot
            patch = xp[:, :, i : i + h, j : j + w_]
            out += np.tensordot(patch, w[:, :, i, j], axes=([1], [1])).transpose(0, 3, 1, 2)
    out += b.reshape(1, c_out, 1, 1)
    return out


def _maxpool2(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """2x2 max-pool stride 2. Returns out and argmax mask for backward."""
    n, c, h, w = x.shape
    assert h % 2 == 0 and w % 2 == 0
    x4 = x.reshape(n, c, h // 2, 2, w // 2, 2)
    x4 = x4.transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h // 2, w // 2, 4)
    idx = np.argmax(x4, axis=-1)
    out = np.max(x4, axis=-1)
    return out, idx


def _maxpool2_backward(dout: np.ndarray, idx: np.ndarray, h: int, w: int) -> np.ndarray:
    n, c, h2, w2 = dout.shape
    dx4 = np.zeros((n, c, h2, w2, 4), dtype=np.float64)
    n_idx, c_idx, i_idx, j_idx = np.indices((n, c, h2, w2))
    dx4[n_idx, c_idx, i_idx, j_idx, idx] = dout
    dx = dx4.reshape(n, c, h2, w2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)
    return dx


class ConvNet28:
    """CNN on 28×28 grayscale — no feature projection, no compact MLP-only head."""

    def __init__(self, n_classes: int = 2, seed: int = 42, fc: int = 64):
        self.n_classes = int(n_classes)
        self.fc = int(fc)
        self.rng = np.random.default_rng(seed)
        # conv1: 1→8
        self.Wc1 = self.rng.normal(0, np.sqrt(2 / 9), (8, 1, 3, 3))
        self.bc1 = np.zeros(8)
        # conv2: 8→16
        self.Wc2 = self.rng.normal(0, np.sqrt(2 / (8 * 9)), (16, 8, 3, 3))
        self.bc2 = np.zeros(16)
        self.flat = 16 * 7 * 7  # 784
        self.Wf1 = self.rng.normal(0, np.sqrt(2 / self.flat), (self.flat, self.fc))
        self.bf1 = np.zeros(self.fc)
        self.Wf2 = self.rng.normal(0, np.sqrt(2 / self.fc), (self.fc, self.n_classes))
        self.bf2 = np.zeros(self.n_classes)

    def _to_nchw(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 4:
            return X
        if X.ndim == 2 and X.shape[1] == 784:
            return X.reshape(-1, 1, 28, 28)
        raise ValueError(f"expected (N,784) or NCHW, got {X.shape}")

    def get_weights(self) -> np.ndarray:
        return np.concatenate(
            [
                self.Wc1.ravel(),
                self.bc1,
                self.Wc2.ravel(),
                self.bc2,
                self.Wf1.ravel(),
                self.bf1,
                self.Wf2.ravel(),
                self.bf2,
            ]
        )

    def set_weights(self, flat: np.ndarray) -> None:
        idx = 0

        def take(shape):
            nonlocal idx
            size = int(np.prod(shape))
            out = flat[idx : idx + size].reshape(shape)
            idx += size
            return out

        self.Wc1 = take(self.Wc1.shape)
        self.bc1 = take(self.bc1.shape)
        self.Wc2 = take(self.Wc2.shape)
        self.bc2 = take(self.bc2.shape)
        self.Wf1 = take(self.Wf1.shape)
        self.bf1 = take(self.bf1.shape)
        self.Wf2 = take(self.Wf2.shape)
        self.bf2 = take(self.bf2.shape)

    def n_params(self) -> int:
        return len(self.get_weights())

    def forward(self, X: np.ndarray):
        x = self._to_nchw(X)
        z1 = _conv2d(x, self.Wc1, self.bc1)
        a1 = relu(z1)
        p1, i1 = _maxpool2(a1)
        z2 = _conv2d(p1, self.Wc2, self.bc2)
        a2 = relu(z2)
        p2, i2 = _maxpool2(a2)
        flat = p2.reshape(p2.shape[0], -1)
        z3 = flat @ self.Wf1 + self.bf1
        a3 = relu(z3)
        z4 = a3 @ self.Wf2 + self.bf2
        a4 = softmax(z4)
        cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
            "p1": p1,
            "i1": i1,
            "z2": z2,
            "a2": a2,
            "p2": p2,
            "i2": i2,
            "flat": flat,
            "z3": z3,
            "a3": a3,
            "z4": z4,
            "a4": a4,
        }
        return a4, cache

    def backward(self, y_onehot: np.ndarray, cache: Dict) -> Dict:
        m = y_onehot.shape[0]
        dz4 = (cache["a4"] - y_onehot) / m
        dWf2 = cache["a3"].T @ dz4
        dbf2 = dz4.sum(axis=0)
        da3 = dz4 @ self.Wf2.T
        dz3 = da3 * relu_deriv(cache["z3"])
        dWf1 = cache["flat"].T @ dz3
        dbf1 = dz3.sum(axis=0)
        dflat = dz3 @ self.Wf1.T
        dp2 = dflat.reshape(cache["p2"].shape)
        da2 = _maxpool2_backward(dp2, cache["i2"], cache["a2"].shape[2], cache["a2"].shape[3])
        dz2 = da2 * relu_deriv(cache["z2"])
        # conv2 grads (finite-diff free: im2col-style accumulate)
        dWc2, dbc2, dp1 = self._conv_backward(cache["p1"], self.Wc2, dz2)
        da1 = _maxpool2_backward(dp1, cache["i1"], cache["a1"].shape[2], cache["a1"].shape[3])
        dz1 = da1 * relu_deriv(cache["z1"])
        dWc1, dbc1, _ = self._conv_backward(cache["x"], self.Wc1, dz1)
        return {
            "dWc1": dWc1,
            "dbc1": dbc1,
            "dWc2": dWc2,
            "dbc2": dbc2,
            "dWf1": dWf1,
            "dbf1": dbf1,
            "dWf2": dWf2,
            "dbf2": dbf2,
        }

    def _conv_backward(
        self, x: np.ndarray, w: np.ndarray, dout: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, c_in, h, ww = x.shape
        c_out, _, k, _ = w.shape
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        dw = np.zeros_like(w)
        dxp = np.zeros_like(xp)
        db = dout.sum(axis=(0, 2, 3))
        for i in range(k):
            for j in range(k):
                patch = xp[:, :, i : i + h, j : j + ww]
                # dout: (N,Cout,H,W), patch: (N,Cin,H,W)
                dw[:, :, i, j] = np.tensordot(dout, patch, axes=([0, 2, 3], [0, 2, 3]))
                # dx contribution
                dxp[:, :, i : i + h, j : j + ww] += np.tensordot(
                    dout, w[:, :, i, j], axes=([1], [0])
                ).transpose(0, 3, 1, 2)
        dx = dxp[:, :, 1:-1, 1:-1]
        return dw, db, dx

    def get_gradient_vector(self, grads: Dict) -> np.ndarray:
        return np.concatenate(
            [
                grads["dWc1"].ravel(),
                grads["dbc1"],
                grads["dWc2"].ravel(),
                grads["dbc2"],
                grads["dWf1"].ravel(),
                grads["dbf1"],
                grads["dWf2"].ravel(),
                grads["dbf2"],
            ]
        )

    def compute_loss(self, y_pred: np.ndarray, y_onehot: np.ndarray) -> float:
        eps = 1e-12
        return float(-np.mean(np.sum(y_onehot * np.log(y_pred + eps), axis=1)))

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01):
        y_onehot = np.zeros((len(y), self.n_classes))
        y_onehot[np.arange(len(y)), y.astype(int)] = 1
        y_pred, cache = self.forward(X)
        loss = self.compute_loss(y_pred, y_onehot)
        grads = self.backward(y_onehot, cache)
        return self.get_gradient_vector(grads), loss

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        y_onehot = np.zeros((len(y), self.n_classes))
        y_onehot[np.arange(len(y)), y.astype(int)] = 1
        y_pred, _ = self.forward(X)
        loss = self.compute_loss(y_pred, y_onehot)
        acc = float(np.mean(np.argmax(y_pred, axis=1) == y))
        return acc, loss
