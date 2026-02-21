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
- `DGM.py` - Deep Galerkin Method neural network implementation
- `train_adjoint_advdiff.py` - Main training script for AFDGM
- `validate_solution.py` - Validation script comparing NN with finite difference
- `requirements.txt` - Python package dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/buchtada/adjointDGM.git
cd adjointDGM

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script to solve the forward and adjoint advection-diffusion equations:

```bash
python train_adjoint_advdiff.py
```

Optional arguments:
```bash
python train_adjoint_advdiff.py \
    --sampling_stages 2000 \
    --steps_per_sample 10 \
    --learning_rate 1e-4 \
    --output_dir ./output/
```

This will:
- Train two neural networks (forward and adjoint)
- Save loss evolution plots
- Save solution fields (forward and adjoint)
- Output training progress to console

**Outputs** (saved to `./output/`):
- `loss_data.npy` - Training loss history
- `forward_solution.npy` - Forward solution u(x,t)
- `adjoint_solution.npy` - Adjoint solution u†(x,t)
- `x_mesh.npy`, `t_mesh.npy` - Spatial and temporal grids
- `loss_evolution.png` - Loss curves during training
- `solutions.png` - Visualization of u and u†

### Validating the Solution

After training, validate the neural network solution against finite difference:

```bash
python validate_solution.py
```

Optional arguments:
```bash
python validate_solution.py \
    --output_dir ./output/ \
    --nx 100 \
    --nt 200
```

This will:
- Solve forward and adjoint PDEs using 4th-order Runge-Kutta
- Compare NN solution with finite difference
- Compute error metrics (L2, max error)
- Perform gradient accuracy assessment via Taylor series test
- Generate validation plots

**Outputs** (saved to `./output/`):
- `validation_comparison.png` - Side-by-side comparison of NN vs FD
- `gradient_accuracy.png` - Taylor series gradient verification

### Quick Start

```bash
# Full workflow
python train_adjoint_advdiff.py --sampling_stages 1000
python validate_solution.py
```

## Documentation

The main manuscript (`doc/manuscript/dgmaf.pdf`) contains:
- Detailed mathematical framework
- Accuracy assessment via Taylor series analysis
- Comparison with finite-difference methods
- Gradient accuracy evaluation

## Algorithm Details

**Neural Network Architecture:**
- LSTM-like layers with gating mechanisms
- 3 hidden layers, 50 nodes per layer
- Xavier initialization
- Adam optimizer with learning rate decay

**Training Strategy:**
- Simultaneous training of forward and adjoint networks
- Random sampling of space-time collocation points
- Periodic resampling to ensure coverage
- Learning rate decay every 500 epochs

## Author

David A. Buchta
Email: buchta1@illinois.edu

## Citation

If you use this work, please cite the manuscript found in `doc/manuscript/dgmaf.pdf`.

## License

Please contact the author for licensing information.
