This repository features a beginner-friendly implementation of a Physics-Informed Neural Network (PINN) designed to solve the Allen–Cahn equation in two spatial dimensions using TensorFlow. The Allen–Cahn equation is a fundamental partial differential equation used to model interface dynamics and phase separation processes in materials and physics.

Instead of relying on labeled data, this PINN approach learns by satisfying the governing physical law directly. The solution begins from a disc-shaped initial condition and evolves over time, with the model using automatic differentiation to compute the required derivatives of the solution.

The notebook includes visualizations that bring clarity to the results. It shows how the full 2D profile changes over time, how the solution evolves along a single 1D slice, and how the φ = 0.5 contour — representing the interface — moves and deforms. These views help develop an intuitive understanding of how the Allen–Cahn equation drives interface behavior.

This project is well-suited for those curious about applying deep learning to physics problems, particularly using neural networks to model PDE-driven dynamics.
