# 🧠 Adaptive Bayesian Classifier (Bell Labs, 1987)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

> **Adaptive Bayesian Classifier** — a faithful Python port of the 1987 Bell Labs algorithm for real-time, confidence-based online learning.  
> This model adapts dynamically to changing data, using a logarithmic feature transform and self-adjusting learning rate for stable, continuous training.

---

## 🔍 Overview
This project implements the **Bell Labs Adaptive Bayesian Classifier**, designed for *online* and *incremental* learning.  
It features:
- Confidence-weighted adaptive updates  
- Selective feature updates (`|x_i| > 1e-10`)  
- `log(1 + x)` nonlinear transform  
- Optional bias term  
- Real-time streaming and incremental learning support  

---

## ⚡ Quick Install

```bash
git clone https://github.com/<your-username>/adaptive-bayes-classifier.git
cd adaptive-bayes-classifier
pip install -r requirements.txt
```

*Only dependency:* `numpy`

---

## 🚀 Quick Start

```python
from adaptive_bayes_classifier import AdaptiveBayesClassifier

clf = AdaptiveBayesClassifier(use_bias=False)
clf.read_data()      # Generates synthetic training data
clf.init_weights()
clf.train()

# Predict
x = [0.2, 0.5, -0.1, 0.7]
pred, p = clf.predict(x)
print(f"Predicted: {pred}, Probability: {p:.3f}")
```

---

## 🧩 Key Features
- **Confidence-Based Learning** – Larger updates for uncertain predictions  
- **Logarithmic Transform** – `log(1 + x)` stabilizes feature magnitudes  
- **Online Learning** – Adapts after every sample  
- **Bias Support** – Toggleable `use_bias` and `update_bias` flags  
- **NumPy Vectorization** – Efficient and lightweight  

---

## 📘 Documentation
Full technical documentation is included in [`adaptive_bayes_docs.md`](./adaptive_bayes_docs.md).

---

## 🧠 Citation
If you use this project in research, please cite:

> Bell Laboratories (1987). *Adaptive Bayesian Classifier: Real-time learning with confidence-weighted updates.*

---

## 🤝 Contributing
Pull requests, bug reports, and extensions (e.g., multi-class or mini-batch variants) are welcome.  
Fork, modify, and open a PR — contributions are actively encouraged.

---

## 📜 License
This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.

---

# Adaptive Bayesian Classifier (Bell Labs, 1987) — Updated Python Port

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

---

## Overview

This implementation mirrors the original **Bell Labs Adaptive Bayesian Classifier (1987)** design, using a confidence-based online learning rule.  
Unlike standard logistic or gradient-descent classifiers, this model adapts its **learning rate dynamically** according to prediction confidence and error magnitude, and selectively updates only non-zero features.  
It’s optimized for **streaming and adaptive learning** tasks where data distributions may drift over time.

---

## 📘 Table of Contents
- [How It Works](#how-it-works)
- [Installation & Requirements](#installation--requirements)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Learning Rule & Parameters](#learning-rule--parameters)
- [Data Format](#data-format)
- [Performance Notes](#performance-notes)
- [Advanced Options](#advanced-options)
- [Conclusion](#conclusion)

---

## ⚙️ How It Works

### Algorithm Summary

1. **Feature transform:**  
   Each feature value is passed through a logarithmic transform:  
   `score = Σ (w_i * log(1 + x_i))`  
   where only features with `|x_i| > 1e-10` are used.

2. **Probability estimation:**  
   The score is mapped to a probability via the logistic sigmoid:  
   `p = 1 / (1 + exp(-score))`

3. **Confidence-based learning:**  
   The adaptive learning rate for each update is:  
   `η = 0.01 * |error| * (1 - |p - 0.5|)`  
   where `error = y - p`.  
   This means:
   - Larger updates when the model is both wrong and uncertain.
   - Smaller updates when it is confident or nearly correct.

4. **Selective feature updates:**  
   Only features with `|x_i| > 1e-10` are updated:  
   `w_i += η * error * x_i`

5. **Optional bias term:**  
   The implementation optionally supports an additive bias `b`:  
   `score = Σ (w_i * log(1 + x_i)) + b`  
   which can be enabled or disabled via the constructor.

---

## 💾 Installation & Requirements

### Requirements
- Python ≥ 3.8  
- NumPy (for vector operations)

### Installation
```bash
pip install numpy
```

Clone or copy the classifier file into your project:
```bash
git clone https://github.com/<your-repo>/adaptive-bayes-classifier.git
```

---

## 🚀 Quick Start

```python
from adaptive_bayes_classifier import AdaptiveBayesClassifier

# Create classifier (no bias for strict Fortran parity)
clf = AdaptiveBayesClassifier(use_bias=False)

# Read or generate data
clf.read_data()

# Initialize weights
clf.init_weights()

# Train online
clf.train()

# Predict
x = [0.2, 0.5, -0.1, 0.7]
pred, p = clf.predict(x)
print(f"Predicted: {pred}, Probability: {p:.3f}")
```

---

## 🧠 API Reference

### Class: `AdaptiveBayesClassifier`

#### Constructor
```python
AdaptiveBayesClassifier(use_bias=False, update_bias=False, lr_scale=0.01, seed=42)
```

| Parameter | Type | Default | Description |
|------------|------|----------|--------------|
| `use_bias` | bool | False | Add a bias term `b` at `w[-1]` |
| `update_bias` | bool | False | Update bias during training (only if `use_bias=True`) |
| `lr_scale` | float | 0.01 | Global scale on the adaptive learning rate |
| `seed` | int | 42 | RNG seed |

#### Methods

##### `read_data(data_file=None)`
Reads data from a text/CSV file or generates synthetic examples.

##### `init_weights()`
Initializes small random weights (`~N(0,0.01)`).

##### `classify(feature_vector)`
Computes probability:  
`P(y=1|x) = sigmoid(Σ (w_i * log(1 + x_i)))`

##### `update_weights(feature_vector, true_label)`
Performs one online update using the adaptive rule.

##### `train(shuffle=True, print_every=100)`
Iterates through samples once, reporting incremental accuracy.

##### `predict(feature_vector)`
Returns `(prediction, probability)`.

---

## 🧩 Usage Examples

### Example 1 — Training on synthetic data
```python
clf = AdaptiveBayesClassifier(use_bias=False)
clf.read_data()      # Generates valid synthetic data automatically
clf.init_weights()
clf.train()
```

### Example 2 — Using with bias
```python
clf = AdaptiveBayesClassifier(use_bias=True, update_bias=True)
clf.read_data()
clf.init_weights()
clf.train()
```

### Example 3 — Streaming data
```python
for x, y in stream:
    p = clf.classify(x)
    clf.update_weights(x, y)
```

---

## 🔬 Learning Rule & Parameters

| Symbol | Description | Default | Notes |
|:-------|:-------------|:---------|:------|
| `η` (eta) | Adaptive learning rate | Computed dynamically | `0.01 * |error| * (1 - |p - 0.5|)` |
| `error` | Difference between true label and predicted probability | — | `y - p` |
| `log1p(x)` | Log transform for feature scaling | — | Requires `x > -1` |
| `EPS` | Feature threshold for updates | `1e-10` | Skip near-zero features |
| `use_bias` | Include additive bias | `False` | Optional |

---

## 📊 Data Format

- **Features:** numeric values, each `> -1` (so `log1p` is defined).  
  Clip data if necessary:
  ```python
  X = np.maximum(X, -1 + 1e-12)
  ```
- **Labels:** binary 0 or 1.
- **Input file:** space- or comma-separated values; last column = label.

Example CSV:
```csv
0.5,0.3,0.8,1
-0.1,0.4,0.2,0
0.9,0.7,0.5,1
```

---

## ⚡ Performance Notes

- **Complexity:** O(n_features) per sample.  
- **Memory:** O(n_features).  
- **Convergence:** naturally slows as confidence increases.  
- **Numerical stability:** `np.log1p()` prevents precision loss for small x.

### Speed & Stability Tips
1. Pre-clip data to `x > -1`.
2. Normalize features if magnitudes vary widely.
3. Use mini-batches if you prefer smoother updates.

---

## 🧮 Advanced Options

### Bias Configuration
| Option | Behavior |
|--------|-----------|
| `use_bias=False` | Strict 1987 Fortran parity |
| `use_bias=True`  | Adds bias term (stored in `w[-1]`) |
| `update_bias=True` | Enables bias weight adaptation |

### Extending to Multi-Class
Use one-vs-all strategy:
```python
class MultiClassAdaptiveBayes:
    def __init__(self, n_classes, **kwargs):
        self.models = [AdaptiveBayesClassifier(**kwargs) for _ in range(n_classes)]

    def train(self, X, y):
        for c in range(self.n_classes):
            yc = (y == c).astype(int)
            m = self.models[c]
            m.X, m.y = X, yc
            m.n_samples, m.n_features = X.shape
            m.init_weights()
            m.train()

    def predict(self, x):
        probs = [m.classify(x) for m in self.models]
        return np.argmax(probs)
```

---

## 🏁 Conclusion

This updated implementation reproduces the **Bell Labs 1987 Adaptive Bayesian Classifier** with its distinctive:
- `log(1+x)` nonlinear feature transform  
- sigmoid probability mapping  
- confidence-driven adaptive learning rate  
- selective weight updates

It provides an interpretable, lightweight framework for **online adaptive classification** that runs in real-time and adapts continuously to new data.

