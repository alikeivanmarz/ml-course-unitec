# Mathematics for Machine Learning

Machine learning rests on a small set of mathematical primitives drawn from linear algebra, calculus, probability, optimization, and information theory. This guide provides working-knowledge refreshers for each area, illustrated with NumPy code rather than proofs. The goal is fluency with the operations that appear in model definitions, loss functions, and training procedures — not mathematical rigour.

**Table of Contents**

1. [Linear Algebra](#1-linear-algebra)
2. [Calculus for Machine Learning](#2-calculus-for-machine-learning)
3. [Probability](#3-probability)
4. [Optimization](#4-optimization)
5. [Information Theory](#5-information-theory)
6. [Math-to-NumPy Translation Table](#6-math-to-numpy-translation-table)
7. [Resources](#7-resources)

---

## 1. Linear Algebra

Linear algebra is the language of features, weights, and transformations. A dataset of `n` samples and `d` features is a matrix in $\mathbb{R}^{n \times d}$; a model's parameters are vectors and matrices; training updates them by linear operations.

### 1.1 Vectors

A vector is a 1-D array of numbers. Vectors of the same shape can be added, subtracted, and scaled.

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

a + b              # element-wise addition
a - b              # element-wise subtraction
2.5 * a            # scalar multiplication
np.dot(a, b)       # dot product -> 32.0
```

### 1.2 Matrices

A matrix is a 2-D array. Shape conventions: `(rows, cols)`. In ML, the design matrix is typically shaped `(n_samples, n_features)`.

```python
A = np.array([[1.0, 2.0],
              [3.0, 4.0]])
A.shape            # (2, 2)
A.T                # transpose -> (2, 2) here, but generally swaps axes
```

### 1.3 Matrix Multiplication

Matrix multiplication composes linear transformations. For `A` of shape `(m, k)` and `B` of shape `(k, n)`, `A @ B` has shape `(m, n)`. The inner dimensions must match.

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 2)
C = A @ B          # shape (3, 2)
```

The dot product is the special case where both operands are 1-D. Element-wise (Hadamard) multiplication is `A * B` and requires broadcasting-compatible shapes.

### 1.4 Norms

A norm measures the size of a vector. The two most common in ML:

| Norm | Definition | NumPy | Used in |
|------|------------|-------|---------|
| L1 | $\sum_i \lvert x_i \rvert$ | `np.linalg.norm(x, 1)` | Lasso regularization, sparsity |
| L2 | $\sqrt{\sum_i x_i^2}$ | `np.linalg.norm(x, 2)` | Ridge regularization, distance metrics |
| Frobenius | matrix L2 of all entries | `np.linalg.norm(M, 'fro')` | Matrix-level regularization |

```python
x = np.array([3.0, 4.0])
np.linalg.norm(x)         # 5.0  (L2 by default)
np.linalg.norm(x, ord=1)  # 7.0
```

### 1.5 Inverse, Rank, Determinant

- **Inverse**: `A_inv` such that `A @ A_inv = I`. Exists only for square, full-rank matrices.
- **Rank**: number of linearly independent rows (or columns). Maximum is `min(rows, cols)`.
- **Determinant**: a scalar; zero indicates singularity (no inverse).

```python
A = np.array([[1.0, 2.0],
              [3.0, 4.0]])

np.linalg.matrix_rank(A)  # 2
np.linalg.det(A)          # -2.0
np.linalg.inv(A)          # explicit inverse — avoid in practice
np.linalg.solve(A, b)     # preferred: solves A x = b numerically
```

### 1.6 Eigendecomposition and SVD

Eigendecomposition factors a square matrix into `A = V diag(λ) V⁻¹`. The eigenvalues `λ` describe how `A` stretches space along directions `V`. SVD generalizes this to any matrix: `A = U Σ Vᵀ`.

```python
A = np.array([[2.0, 1.0],
              [1.0, 3.0]])

eigvals, eigvecs = np.linalg.eig(A)
U, S, Vt = np.linalg.svd(A)
```

SVD underpins PCA, low-rank approximation, and recommender systems. Eigendecomposition appears in spectral clustering and covariance analysis.

### 1.7 Where Linear Algebra Appears in ML

| Concept | Use |
|---------|-----|
| Matrix multiplication | Forward pass of every linear and dense layer |
| L1 / L2 norms | Lasso, Ridge, weight regularization |
| Eigendecomposition | PCA, spectral methods |
| SVD | Low-rank approximation, latent factor models |
| Determinant | Multivariate Gaussians, change of variables |

---

## 2. Calculus for Machine Learning

Training is the process of adjusting parameters to reduce a loss. Calculus describes how the loss changes as the parameters change — the information used by every gradient-based optimizer.

### 2.1 Derivatives

The derivative of a scalar function $f(x)$ at a point $x$ is the local rate of change. For $f(x) = x^2$, $f'(x) = 2x$.

```python
def f(x):
    return x ** 2

x0 = 3.0
h = 1e-5
numerical = (f(x0 + h) - f(x0 - h)) / (2 * h)  # central difference
analytical = 2 * x0
# both ≈ 6.0
```

### 2.2 Partial Derivatives and Gradients

For a function of several variables $f(x_1, x_2, \ldots, x_n)$, the partial derivative with respect to $x_i$ holds the others fixed. The gradient $\nabla f$ stacks all partials into a vector.

For $f(x, y) = x^2 + 3xy + y^2$:
- $\partial f / \partial x = 2x + 3y$
- $\partial f / \partial y = 3x + 2y$
- $\nabla f = [2x + 3y,\; 3x + 2y]$

The gradient points in the direction of steepest increase. Optimization moves opposite to it.

### 2.3 The Chain Rule

For composed functions $f(g(x))$, the derivative is $f'(g(x)) \cdot g'(x)$. Backpropagation is the repeated application of the chain rule across a network's layers, propagating gradients from the loss back to each parameter.

### 2.4 Jacobians and Hessians

- **Jacobian**: the matrix of all first-order partial derivatives of a vector-valued function. Shape `(output_dim, input_dim)`.
- **Hessian**: the matrix of second-order partial derivatives of a scalar function. Shape `(input_dim, input_dim)`. Captures curvature.

Second-order methods (Newton's method, L-BFGS) use the Hessian or its approximation; first-order methods (SGD, Adam) use only the gradient.

### 2.5 Where Calculus Appears in ML

| Concept | Use |
|---------|-----|
| Gradient | Every gradient-based optimizer |
| Chain rule | Backpropagation through layers |
| Jacobian | Vector-valued layer derivatives |
| Hessian | Newton-type optimizers, curvature analysis |

---

## 3. Probability

Probability quantifies uncertainty. Many ML models output probabilities (logistic regression, softmax classifiers, Bayesian models), and many loss functions are derived from likelihoods.

### 3.1 Random Variables and Distributions

A random variable maps outcomes to numbers. Distributions describe how likely each outcome is.

| Distribution | Support | Common use |
|--------------|---------|------------|
| Bernoulli | {0, 1} | Single binary outcome |
| Binomial | {0, 1, …, n} | Count of successes in n trials |
| Categorical | {1, …, K} | Single multi-class outcome |
| Normal (Gaussian) | $\mathbb{R}$ | Continuous noise, weight initialization |
| Uniform | [a, b] | Random initialization, sampling |
| Exponential | $\mathbb{R}_{\geq 0}$ | Waiting times, survival models |

```python
from scipy import stats

# Sample 1000 values from a standard normal
samples = stats.norm.rvs(loc=0, scale=1, size=1000)

# Probability density at x = 1.5
stats.norm.pdf(1.5, loc=0, scale=1)

# Cumulative probability P(X <= 1.5)
stats.norm.cdf(1.5, loc=0, scale=1)
```

### 3.2 Expectation and Variance

The expectation $\mathbb{E}[X]$ is the mean of a random variable's distribution; the variance $\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$ measures spread.

```python
x = np.random.normal(loc=2.0, scale=3.0, size=10000)
x.mean()    # ≈ 2.0
x.var()     # ≈ 9.0
x.std()     # ≈ 3.0
```

### 3.3 Conditional Probability and Bayes' Theorem

The conditional probability $P(A \mid B)$ is the probability of $A$ given that $B$ occurred. Bayes' theorem inverts the conditioning:

$$P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}$$

In ML, $A$ is often a class label and $B$ is the observed features. The posterior $P(A \mid B)$ is what classifiers estimate; naive Bayes models compute it directly under independence assumptions.

### 3.4 Where Probability Appears in ML

| Concept | Use |
|---------|-----|
| Likelihood | Maximum-likelihood estimation, loss derivation |
| Bayes' theorem | Naive Bayes, Bayesian inference, posterior estimation |
| Gaussian distribution | Weight initialization, kernel density estimation |
| Expectation | Loss functions are expected losses over the data distribution |

---

## 4. Optimization

Training is optimization: find parameters that minimize a loss function. Most ML optimizers are first-order methods that follow the negative gradient.

### 4.1 Gradient Descent

Update parameters $\theta$ by stepping opposite to the gradient of the loss $L$:

$$\theta \leftarrow \theta - \eta \, \nabla_\theta L$$

where $\eta$ is the learning rate.

```python
# Minimize f(x) = (x - 3) ** 2
def grad(x):
    return 2 * (x - 3)

x = 0.0
lr = 0.1
for _ in range(100):
    x -= lr * grad(x)
# x ≈ 3.0
```

### 4.2 Stochastic and Mini-Batch Gradient Descent

Computing the gradient over the entire dataset (batch GD) is expensive. SGD uses one sample per step; mini-batch GD uses a small batch (typical sizes: 32–512). Mini-batch GD is the default in deep learning — it balances gradient noise against compute cost and gives noisy steps that help escape shallow minima.

### 4.3 Momentum and Adaptive Methods

Vanilla SGD oscillates in narrow valleys. Variants modify the update rule:

| Optimizer | Idea | Notes |
|-----------|------|-------|
| Momentum | Accumulate a velocity vector | Smooths updates, accelerates in consistent directions |
| RMSProp | Per-parameter learning-rate scaling by recent gradient magnitude | Handles features at different scales |
| Adam | Combines momentum and RMSProp | Default choice for deep nets; usually robust |
| L-BFGS | Approximates second-order curvature | Strong on small problems; rare in deep learning |

### 4.4 Learning Rate

The single most influential hyperparameter. Too high: divergence. Too low: slow or stuck convergence. Common patterns:

- **Constant**: simplest; works with adaptive optimizers like Adam.
- **Step decay**: reduce by a factor at fixed epochs.
- **Cosine schedule**: smoothly decay from initial to near-zero.
- **Warm-up + decay**: ramp up over a few epochs, then decay. Common in transformer training.

---

## 5. Information Theory

Information-theoretic quantities appear in loss functions (cross-entropy), feature selection (mutual information), and model comparison (KL divergence).

### 5.1 Entropy

Entropy measures the uncertainty of a distribution. For a discrete distribution $p$ over $K$ outcomes:

$$H(p) = -\sum_{k=1}^{K} p_k \log p_k$$

A uniform distribution has maximum entropy; a one-hot distribution has zero.

```python
from scipy.stats import entropy

p_uniform = [0.25, 0.25, 0.25, 0.25]
p_skewed  = [0.97, 0.01, 0.01, 0.01]

entropy(p_uniform)  # ≈ 1.386 (in nats)
entropy(p_skewed)   # ≈ 0.176
```

### 5.2 Cross-Entropy and KL Divergence

**Cross-entropy** between true distribution $p$ and predicted $q$:

$$H(p, q) = -\sum_k p_k \log q_k$$

**KL divergence**: cross-entropy minus entropy. Measures how far $q$ is from $p$:

$$D_{\mathrm{KL}}(p \,\|\, q) = H(p, q) - H(p)$$

Cross-entropy is the standard loss for classification because minimizing it (with $p$ a one-hot label) is equivalent to maximum likelihood under the model.

### 5.3 Mutual Information

Mutual information $I(X; Y)$ measures how much knowing $X$ reduces uncertainty about $Y$. It is symmetric and non-negative; zero means $X$ and $Y$ are independent.

In feature selection, mutual information ranks features by how informative they are about the target — capturing non-linear dependencies that correlation misses.

---

## 6. Math-to-NumPy Translation Table

| Mathematical notation | NumPy expression |
|-----------------------|------------------|
| $\mathbf{a} \cdot \mathbf{b}$ | `a @ b` or `np.dot(a, b)` |
| $A \mathbf{x}$ | `A @ x` |
| $A B$ | `A @ B` |
| $A^\top$ | `A.T` |
| $A^{-1} \mathbf{b}$ | `np.linalg.solve(A, b)` |
| $\lVert \mathbf{x} \rVert_2$ | `np.linalg.norm(x)` |
| $\sum_i x_i$ | `x.sum()` |
| $\prod_i x_i$ | `x.prod()` |
| $\mathbb{E}[X]$ (sample) | `x.mean()` |
| $\mathrm{Var}(X)$ (sample) | `x.var()` |
| $\exp(\mathbf{x})$ | `np.exp(x)` |
| $\log(\mathbf{x})$ | `np.log(x)` |
| Softmax of $\mathbf{x}$ | `np.exp(x) / np.exp(x).sum()` (numerically: subtract `x.max()` first) |

---

## 7. Resources

- [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) — visual treatment of vectors, matrices, eigenvectors, and SVD.
- [3Blue1Brown — Essence of Calculus](https://www.3blue1brown.com/topics/calculus) — derivatives, integrals, and the chain rule.
- [Mathematics for Machine Learning (Deisenroth, Faisal, Ong)](https://mml-book.github.io/) — open-access textbook covering linear algebra, calculus, probability, and the math behind core ML methods.
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — comprehensive reference for matrix identities and derivatives.
- [Goodfellow, Bengio, Courville — *Deep Learning*, Part I](https://www.deeplearningbook.org/) — chapters 2–4 cover applied linear algebra, probability, and numerical computation.
- [NumPy linear algebra reference](https://numpy.org/doc/stable/reference/routines.linalg.html) — operator and function listings.
