# 🧠 Physics-Informed Neural Network for the Allen–Cahn Equation (2D) of Phase-field Modelling

This repository provides a beginner-friendly implementation of a Physics-Informed Neural Network (PINN) for solving the Allen–Cahn equation in two spatial dimensions, using TensorFlow.

The Allen–Cahn equation is a foundational model in **phase-field modeling**, widely used to describe the evolution of interfaces in systems with a **non-conserved order parameter**. It captures how phase boundaries move and shrink over time due to interfacial energy, making it especially relevant in materials science, image processing, and pattern formation studies.

Unlike traditional numerical solvers, this approach uses a neural network trained to satisfy the underlying partial differential equation directly, without requiring labeled data. The initial condition is defined as a smooth disc-like profile, and the model evolves it through time while respecting the physics embedded in the Allen–Cahn dynamics.

To help visualize what the network learns, the notebook includes views of the full 2D field, time evolution along a fixed 1D slice, and the motion of the φ = 0.5 contour, which represents the interface position. These tools help develop intuition about interface dynamics in phase-field systems.

The code is well-suited for those beginning their exploration of physics-informed machine learning, as well as researchers curious about using neural networks for modeling phase-field phenomena.
