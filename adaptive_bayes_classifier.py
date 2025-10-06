#!/usr/bin/env python3
"""
ADAPTIVE BAYESIAN CLASSIFIER - Bell Labs 1987 (Python port)
Author: <you>
Notes:
- This mirrors the shared Fortran behavior:
    score = sum_i w_i * log(1 + x_i) over |x_i| > 1e-10
    prob  = sigmoid(score)
    error = y - prob
    learning_rate = 0.01 * |error| * (1 - |prob - 0.5|)
    For |x_i| > 1e-10: w_i += learning_rate * error * x_i
- Optional bias can be enabled with use_bias=True. If enabled, bias is the last
  element of w and (optionally) updated (controlled by update_bias flag).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Tuple, List

# ------------------- Constants (Fortran-style) -------------------
MAXFEAT = 50
MAXSAMP = 100_000

# Threshold used in the original Fortran code
EPS = 1e-10

# ------------------- Utilities -------------------
def stable_sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def clip_for_log1p(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Ensure inputs are > -1 for log1p. If any x <= -1, clip to (-1 + eps).
    """
    return np.maximum(x, -1.0 + eps)

# ------------------- Classifier -------------------
class AdaptiveBayesClassifier:
    """
    Online learner matching the provided Fortran subroutines.

    Parameters
    ----------
    use_bias : bool
        If True, maintain an explicit bias term at w[-1] and include it in the score.
    update_bias : bool
        If True and use_bias is True, update the bias each step with the same LR*error rule.
        (Bias update did not exist in the Fortran snippet; default False keeps parity.)
    lr_scale : float
        Multiplier on the confidence-based learning rate (defaults to 0.01 per snippet).
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        use_bias: bool = False,
        update_bias: bool = False,
        lr_scale: float = 0.01,
        seed: int = 42,
    ):
        self.use_bias = use_bias
        self.update_bias = update_bias and use_bias
        self.lr_scale = lr_scale

        self.X: Optional[np.ndarray] = None  # (n_samples, n_features)
        self.y: Optional[np.ndarray] = None  # (n_samples,)
        self.w: Optional[np.ndarray] = None  # (n_features [+1 if bias],)
        self.n_samples: int = 0
        self.n_features: int = 0

        self.rng = np.random.default_rng(seed)

    # ---------- Data I/O ----------
    def read_data(self, data_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Read data or generate synthetic.
        File format: space- or comma-separated, last column = label (0/1).
        """
        if data_file:
            feats: List[List[float]] = []
            labs: List[int] = []
            with open(data_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p for p in line.replace(",", " ").split() if p]
                    *fv, lab = parts
                    fv = list(map(float, fv))
                    lab = int(lab)
                    feats.append(fv)
                    labs.append(lab)
            X = np.asarray(feats, dtype=np.float64)
            y = np.asarray(labs, dtype=np.int32)
        else:
            # Synthetic data: ensure features > -1 so log1p is defined
            self.n_samples = min(1000, MAXSAMP)
            self.n_features = min(10, MAXFEAT)
            # Draw features from Normal, then shift/clip so they’re mostly valid for log1p
            X = self.rng.normal(loc=0.0, scale=1.0, size=(self.n_samples, self.n_features))
            X = clip_for_log1p(X, eps=1e-8)

            # Create labels from a random linear rule in the log1p space
            true_w = self.rng.normal(scale=1.0, size=self.n_features)
            logits = (np.log1p(X) @ true_w)  # no bias by default
            p = 1.0 / (1.0 + np.exp(-logits))
            y = (self.rng.random(self.n_samples) < p).astype(np.int32)

        # Enforce limits
        n_samples, n_features = X.shape
        if n_features > MAXFEAT:
            X = X[:, :MAXFEAT]
            n_features = MAXFEAT
        if n_samples > MAXSAMP:
            X = X[:MAXSAMP]
            y = y[:MAXSAMP]
            n_samples = MAXSAMP

        # Store
        self.X, self.y = X, y
        self.n_samples, self.n_features = n_samples, n_features
        return self.X, self.y, self.n_samples, self.n_features

    def init_weights(self) -> np.ndarray:
        if self.n_features <= 0:
            raise ValueError("Call read_data() before init_weights().")
        w_dim = self.n_features + (1 if self.use_bias else 0)
        # small random init
        self.w = self.rng.normal(scale=0.01, size=w_dim).astype(np.float64)
        return self.w

    # ---------- Core math (Fortran parity) ----------
    def _classify_no_bias(self, x: np.ndarray) -> float:
        """
        PROB = sigmoid( sum_i w_i * log(1 + x_i), for |x_i| > EPS )
        """
        if self.w is None:
            raise ValueError("Weights not initialized.")
        if x.shape[0] != self.n_features or self.w.shape[0] != self.n_features:
            raise ValueError("Feature/weight dimension mismatch for no-bias mode.")
        x = clip_for_log1p(x)  # safety for log1p
        mask = np.abs(x) > EPS
        score = float(np.dot(self.w[mask], np.log1p(x[mask])))
        return stable_sigmoid(score)

    def _update_no_bias(self, x: np.ndarray, y_true: int) -> None:
        """
        LEARNING_RATE = 0.01 * |error| * (1 - |prob - 0.5|)
        For |x_i| > EPS: w_i += LR * error * x_i
        """
        assert self.w is not None
        p = self._classify_no_bias(x)
        error = float(y_true) - p
        lr = self.lr_scale * abs(error) * (1.0 - abs(p - 0.5))
        mask = np.abs(x) > EPS
        self.w[mask] += lr * error * x[mask]

    def _classify_with_bias(self, x: np.ndarray) -> float:
        """
        w = [w_1..w_n, b]; score = sum_i w_i*log(1+x_i) + b
        """
        if self.w is None:
            raise ValueError("Weights not initialized.")
        if self.w.shape[0] != self.n_features + 1:
            raise ValueError("Feature/weight dimension mismatch for bias mode.")
        x = clip_for_log1p(x)
        mask = np.abs(x) > EPS
        score = float(np.dot(self.w[:-1][mask], np.log1p(x[mask])) + self.w[-1])
        return stable_sigmoid(score)

    def _update_with_bias(self, x: np.ndarray, y_true: int) -> None:
        """
        Same LR rule; optionally update bias with lr*error.
        """
        assert self.w is not None
        p = self._classify_with_bias(x)
        error = float(y_true) - p
        lr = self.lr_scale * abs(error) * (1.0 - abs(p - 0.5))

        mask = np.abs(x) > EPS
        self.w[:-1][mask] += lr * error * x[mask]
        if self.update_bias:
            self.w[-1] += lr * error

    # ---------- Public API ----------
    def classify(self, feature_vector: np.ndarray) -> float:
        """Return probability P(y=1|x)."""
        x = np.asarray(feature_vector, dtype=np.float64)
        if self.use_bias:
            return self._classify_with_bias(x)
        else:
            return self._classify_no_bias(x)

    def update_weights(self, feature_vector: np.ndarray, true_label: int) -> None:
        """Apply one online update step."""
        x = np.asarray(feature_vector, dtype=np.float64)
        if self.use_bias:
            self._update_with_bias(x, int(true_label))
        else:
            self._update_no_bias(x, int(true_label))

    def train(self, shuffle: bool = True, print_every: int = 100) -> float:
        """Stream through the dataset once (online learning). Returns stream accuracy."""
        if self.X is None or self.y is None:
            raise ValueError("Call read_data() first.")
        if self.w is None:
            self.init_weights()

        idx = np.arange(self.n_samples)
        if shuffle:
            self.rng.shuffle(idx)

        correct = 0
        for i, j in enumerate(idx, start=1):
            xj = self.X[j]
            yj = int(self.y[j])
            p = self.classify(xj)
            pred = 1 if p >= 0.5 else 0
            correct += (pred == yj)
            self.update_weights(xj, yj)

            if print_every and (i % print_every == 0):
                acc = correct / i
                print(f"Processed {i}/{self.n_samples}  |  Acc: {acc:.3f}")

        final_acc = correct / self.n_samples
        print("\nTraining complete.")
        print(f"Final stream accuracy: {final_acc:.3f}")
        print(f"Weights (first 6): {self.w[:6]}")
        return final_acc

    def predict(self, feature_vector: np.ndarray) -> Tuple[int, float]:
        p = self.classify(feature_vector)
        return (1 if p >= 0.5 else 0), p

# ------------------- Demo main -------------------
def main():
    # Toggle bias behavior here
    clf = AdaptiveBayesClassifier(use_bias=False, update_bias=False, lr_scale=0.01, seed=123)

    clf.read_data()      # or clf.read_data("path/to/data.txt")
    clf.init_weights()
    clf.train()

    print("\n--- Testing on new samples ---")
    for i in range(5):
        x = clip_for_log1p(clf.rng.normal(size=clf.n_features))
        pred, p = clf.predict(x)
        print(f"Sample {i+1}: pred={pred}, p={p:.3f}")

if __name__ == "__main__":
    main()
