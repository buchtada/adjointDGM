# Adjoint-Forward Deep Galerkin Method (AFDGM)

A novel approach for solving partial differential equations (PDEs) using deep neural networks combined with adjoint-based sensitivity analysis.

## Overview

The **Adjoint-Forward Deep Galerkin Method (AFDGM)** approximates both forward and adjoint solutions to PDEs using deep neural networks. This method addresses the computational challenge of computing adjoint states for optimization and sensitivity analysis without the memory burden of storing entire forward state trajectories.

## Problem Statement

Computing adjoint states is essential for:
- Optimization
- Sensitivity analysis
- Data assimilation
- Experimental design

However, traditional methods require access to the entire trajectory of the forward state, which is computationally expensive. By approximating both forward and adjoint PDEs with neural networks, AFDGM avoids these memory and computational bottlenecks.

## Methodology

The Deep Galerkin Method is used to train neural networks that minimize the residual of both:
- Forward PDE and boundary conditions
- Adjoint PDE and boundary conditions

The loss function combines six terms:
- Forward PDE residual (L_PDE)
- Forward initial conditions (L_IC)
- Forward boundary conditions (L_BC)
- Adjoint PDE residual (L†_PDE)
- Adjoint initial conditions (L†_IC)
- Adjoint boundary conditions (L†_BC)

## Test Case

The method is demonstrated on the **advection-diffusion equation**:

```
∂u/∂t + c∂u/∂x - ν∂²u/∂x² = 0
```

Parameters:
- Advection speed: c = 1.0
- Diffusion coefficient: ν = 0.01
- Initial condition: Gaussian profile

The objective is to identify initial perturbations that maximize final state energy.

## Repository Contents

- `doc/manuscript/` - LaTeX manuscript and compiled PDF
- `doc/manuscript/figures/` - Publication-ready figures
- `doc/references.bib` - Bibliography with 50+ academic references

## Documentation

The main manuscript (`doc/manuscript/dgmaf.pdf`) contains:
- Detailed mathematical framework
- Accuracy assessment via Taylor series analysis
- Comparison with finite-difference methods
- Gradient accuracy evaluation

## Author

David A. Buchta
Email: buchta1@illinois.edu

## Citation

If you use this work, please cite the manuscript found in `doc/manuscript/dgmaf.pdf`.

## License

Please contact the author for licensing information.
