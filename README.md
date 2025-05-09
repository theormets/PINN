# 🧠 Physics-Informed Neural Network for the Allen–Cahn Equation (2D)

This repository presents a Physics-Informed Neural Network (PINN) implementation using TensorFlow to solve the Allen–Cahn equation in two spatial dimensions. The code demonstrates how neural networks can be trained not on labeled data, but by learning to obey a physical law described by a partial differential equation (PDE).

## 🧪 What is the Allen–Cahn Equation?

The Allen–Cahn equation describes the motion of phase boundaries in systems where the order parameter is non-conserved. It is widely used in phase-field modeling for simulating interface evolution in materials science, pattern formation, and other applications.

The 2D Allen–Cahn equation is given by:

```
∂u/∂t = ε² ∇²u - 5u³ + 5u
```

where:
- `u(x, y, t)` is the order parameter (e.g., phase or concentration),
- `ε` controls the interface width,
- `∇²u` is the Laplacian (spatial second derivative),
- The nonlinear terms `-5u³ + 5u` drive the reaction kinetics.

In this setup, the solution evolves from an initial disc-shaped profile and dissipates over time, governed by curvature and energetics.

## 🔧 How the PINN Works

Instead of using training data, a PINN learns the solution by minimizing a physics-based loss function that enforces the PDE and known conditions.

The loss function `L_total` is composed of the following components:

PDE Residual Loss measures how well the neural network satisfies the Allen–Cahn equation at collocation points across space and time:

```
L_pde = (1 / N_f) * Σ [ ∂u/∂t - ε² (∂²u/∂x² + ∂²u/∂y²) + 5u³ - 5u ]²
```

Initial Condition Loss ensures that the predicted solution at time `t = 0` matches a known disc-shaped profile:

```
L_ic = (1 / N_u) * Σ [ u_pred(x, y, t=0) - u_initial(x, y) ]²
```

L2 regularization is applied to all network weights to prevent overfitting:

```
L_reg = λ * Σ ||W||²
```

The final loss function optimized during training is:

```
L_total = L_pde + α * L_ic + λ * L_reg
```

Here, `α = 2.0` emphasizes the importance of the initial condition early in training, and `λ = 1e-6` applies light regularization.

## 📊 Visualizations

The notebook includes several visual tools to understand the model’s predictions. It shows the full 2D evolution of the field `u(x, y, t)`, the interface profile along a horizontal slice at `y = 0`, and the motion of the φ = 0.5 level set over time to track interface changes.

## 🚀 Run on Google Colab

You can open and run the notebook in your browser with no installation:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/<your-notebook>.ipynb)

## 🧑‍🎓 Who is this for?

This project is ideal for students, researchers, and beginners exploring physics-informed machine learning, phase-field modeling, or deep learning for PDEs.
