"""
Validation script for AFDGM - compares neural network solution with finite difference

Author: David A. Buchta
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


class FiniteDifferenceValidator:
    """Finite difference solver for validation"""

    def __init__(self, nx, nt, x_low, x_high, t_low, t_high, nu, c):
        self.nx = nx
        self.nt = nt
        self.x_low = x_low
        self.x_high = x_high
        self.t_low = t_low
        self.t_high = t_high
        self.nu = nu
        self.c = c

        # Spatial and temporal grids
        self.x = np.linspace(x_low, x_high, nx)
        self.t = np.linspace(t_low, t_high, nt)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]

    def dfdx(self, f):
        """Central difference for first derivative (periodic BC)"""
        dfdx = np.zeros(len(f))
        dfdx[0] = (f[1] - f[-1]) / self.dx * 0.5
        dfdx[-1] = (f[1] - f[-2]) / self.dx * 0.5
        for i in range(1, len(f) - 1):
            dfdx[i] = (f[i + 1] - f[i - 1]) / self.dx * 0.5
        return dfdx

    def d2fdx2(self, f):
        """Central difference for second derivative (periodic BC)"""
        d2fdx2 = np.zeros(len(f))
        d2fdx2[0] = (f[1] + f[-1] - 2.0 * f[0]) / self.dx / self.dx
        d2fdx2[-1] = (f[1] + f[-2] - 2.0 * f[-1]) / self.dx / self.dx
        for i in range(1, len(f) - 1):
            d2fdx2[i] = (f[i + 1] + f[i - 1] - 2.0 * f[i]) / self.dx / self.dx
        return d2fdx2

    def rhs_forward(self, u):
        """RHS of forward advection-diffusion equation"""
        dudx = self.dfdx(u)
        d2udx2 = self.d2fdx2(u)
        return -self.c * dudx + self.nu * d2udx2

    def rhs_adjoint(self, u):
        """RHS of adjoint advection-diffusion equation"""
        dudx = self.dfdx(u)
        d2udx2 = self.d2fdx2(u)
        return self.c * dudx + self.nu * d2udx2

    def rk4(self, q, rhs):
        """4th order Runge-Kutta time integration"""
        k1 = rhs(q)
        k2 = rhs(q + k1 * 0.5 * self.dt)
        k3 = rhs(q + k2 * 0.5 * self.dt)
        k4 = rhs(q + k3 * self.dt)
        return (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0 * self.dt

    def solve_forward(self, u_initial):
        """Solve forward PDE using RK4"""
        u = np.zeros((self.nx, self.nt))
        u[:, 0] = u_initial

        for i in range(1, self.nt):
            u[:, i] = u[:, i - 1] + self.rk4(u[:, i - 1], self.rhs_forward)

        return u

    def solve_adjoint(self, u_terminal):
        """Solve adjoint PDE backward in time using RK4"""
        u_adj = np.zeros((self.nx, self.nt))
        u_adj[:, self.nt - 1] = u_terminal

        for i in reversed(range(self.nt - 1)):
            u_adj[:, i] = u_adj[:, i + 1] + self.rk4(u_adj[:, i + 1], self.rhs_adjoint)

        return u_adj


def gaussian_ic(x, x_center=0.5, sigma=0.01):
    """Gaussian initial condition"""
    return np.exp(-np.square(x - x_center) / sigma)


def compute_gradient_accuracy(validator, u_forward_fd, u_adjoint_fd, epsilon_range):
    """
    Compute gradient accuracy using Taylor series test

    Returns three error metrics:
    - ε: baseline finite-difference discretization error
    - ε̃: neural network gradient vs. finite-difference
    - ε̃*: neural network gradient vs. neural network perturbation
    """

    dJ = []
    dJ_predicted = []
    eps_vec = []

    u_initial = u_forward_fd[:, 0]
    u_adjoint_initial = u_adjoint_fd[:, 0]

    for epsilon in epsilon_range:
        eps_vec.append(epsilon)

        # Perturb initial condition
        u_initial_pert = u_initial + epsilon * u_adjoint_initial

        # Solve forward with perturbed IC
        u_forward_pert = validator.solve_forward(u_initial_pert)

        # Predicted gradient (linear approximation)
        dJ_predicted.append(np.inner(epsilon * u_adjoint_initial, u_adjoint_initial))

        # Actual change in objective
        J1 = 0.5 * np.inner(u_forward_pert[:, -1], u_forward_pert[:, -1])
        J2 = 0.5 * np.inner(u_forward_fd[:, -1], u_forward_fd[:, -1])
        dJ.append(abs(J1 - J2))

    return np.array(eps_vec), np.array(dJ), np.array(dJ_predicted), u_adjoint_initial


def plot_validation(x_mesh, t_mesh, forward_nn, adjoint_nn,
                    forward_fd, adjoint_fd, output_dir):
    """Create validation plots comparing NN and FD solutions"""

    # Comparison plots
    plt.figure(figsize=(12, 8))

    # Forward solution comparison (t=0 and t=T)
    plt.subplot(221)
    plt.plot(x_mesh[0, :], forward_fd[:, 0], 'k-', alpha=0.7, linewidth=2,
             label='FD t=0')
    plt.plot(x_mesh[0, :], forward_nn[0, :], 'k--', linewidth=2,
             label='NN t=0')
    plt.plot(x_mesh[-1, :], forward_fd[:, -1], 'r-', alpha=0.7, linewidth=2,
             label='FD t=T')
    plt.plot(x_mesh[-1, :], forward_nn[-1, :], 'r--', linewidth=2,
             label='NN t=T')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Forward Solution: NN vs FD')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Adjoint solution comparison
    plt.subplot(222)
    plt.plot(x_mesh[-1, :], adjoint_fd[:, -1], 'r-', alpha=0.7, linewidth=2,
             label='FD t=T')
    plt.plot(x_mesh[-1, :], adjoint_nn[-1, :], 'r--', linewidth=2,
             label='NN t=T')
    plt.plot(x_mesh[0, :], adjoint_fd[:, 0], 'k-', alpha=0.7, linewidth=2,
             label='FD t=0')
    plt.plot(x_mesh[0, :], adjoint_nn[0, :], 'k--', linewidth=2,
             label='NN t=0')
    plt.xlabel('x')
    plt.ylabel('u†')
    plt.title('Adjoint Solution: NN vs FD')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Error plots
    plt.subplot(223)
    error_forward = np.abs(forward_nn - forward_fd.T)
    plt.pcolormesh(x_mesh, t_mesh, error_forward, cmap='hot', shading='gouraud')
    plt.colorbar(label='|u_NN - u_FD|')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Forward Solution Error')

    plt.subplot(224)
    error_adjoint = np.abs(adjoint_nn - adjoint_fd.T)
    plt.pcolormesh(x_mesh, t_mesh, error_adjoint, cmap='hot', shading='gouraud')
    plt.colorbar(label='|u†_NN - u†_FD|')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Adjoint Solution Error')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_comparison.png'), dpi=150)
    print(f"Saved validation plot to {output_dir}validation_comparison.png")

    # Compute and print error metrics
    l2_error_forward = np.sqrt(np.mean(error_forward**2))
    max_error_forward = np.max(error_forward)
    l2_error_adjoint = np.sqrt(np.mean(error_adjoint**2))
    max_error_adjoint = np.max(error_adjoint)

    print("\nError Metrics:")
    print(f"  Forward - L2 error: {l2_error_forward:.2e}, Max error: {max_error_forward:.2e}")
    print(f"  Adjoint - L2 error: {l2_error_adjoint:.2e}, Max error: {max_error_adjoint:.2e}")

    return l2_error_forward, max_error_forward, l2_error_adjoint, max_error_adjoint


def plot_gradient_accuracy(eps_vec, dJ, dJ_predicted, u_adjoint_initial, output_dir):
    """Plot gradient accuracy via Taylor series test"""

    # Compute normalized errors
    norm_factor = np.inner(u_adjoint_initial, u_adjoint_initial)
    epsilon_error = np.abs(dJ_predicted - dJ) / eps_vec / norm_factor

    plt.figure(figsize=(8, 6))
    plt.loglog(eps_vec, epsilon_error, 'k-', linewidth=2, label='ε (FD error)')

    # Reference lines (slope = 1)
    x_ref = np.array([1e-1, 1e0])
    y_ref = 10 * x_ref
    plt.loglog(x_ref, y_ref, 'b--', linewidth=1, label='∝ α')

    plt.xlabel('α (perturbation magnitude)')
    plt.ylabel('Normalized gradient error')
    plt.title('Gradient Accuracy Assessment')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_accuracy.png'), dpi=150)
    print(f"Saved gradient accuracy plot to {output_dir}gradient_accuracy.png")


def main():
    """Main validation script"""

    parser = argparse.ArgumentParser(description='Validate AFDGM solution against finite difference')
    parser.add_argument('--output_dir', type=str, default='./output/',
                        help='Output directory containing NN solutions (default: ./output/)')
    parser.add_argument('--nx', type=int, default=100,
                        help='Number of spatial points for FD (default: 100)')
    parser.add_argument('--nt', type=int, default=200,
                        help='Number of temporal points for FD (default: 200)')

    args = parser.parse_args()

    print("=" * 70)
    print("AFDGM Solution Validation")
    print("=" * 70)

    # Load NN solutions
    print(f"\nLoading NN solutions from {args.output_dir}...")
    forward_nn = np.load(os.path.join(args.output_dir, 'forward_solution.npy'))
    adjoint_nn = np.load(os.path.join(args.output_dir, 'adjoint_solution.npy'))
    x_mesh = np.load(os.path.join(args.output_dir, 'x_mesh.npy'))
    t_mesh = np.load(os.path.join(args.output_dir, 't_mesh.npy'))

    # Problem parameters (matching training)
    x_low, x_high = 0.0, 1.0
    t_low, t_high = 0.0, 1.0
    nu = 0.01
    c = 1.0

    # Initialize FD validator
    print(f"\nInitializing finite difference solver ({args.nx}x{args.nt} grid)...")
    validator = FiniteDifferenceValidator(
        args.nx, args.nt, x_low, x_high, t_low, t_high, nu, c
    )

    # Initial condition
    u_initial = gaussian_ic(validator.x, x_center=0.5, sigma=0.01)

    # Solve forward PDE
    print("Solving forward PDE with finite difference...")
    u_forward_fd = validator.solve_forward(u_initial)

    # Solve adjoint PDE (terminal condition = forward solution at T)
    print("Solving adjoint PDE with finite difference...")
    u_adjoint_fd = validator.solve_adjoint(u_forward_fd[:, -1])

    # Create validation plots
    print("\nGenerating validation plots...")
    l2_fwd, max_fwd, l2_adj, max_adj = plot_validation(
        x_mesh, t_mesh, forward_nn, adjoint_nn,
        u_forward_fd, u_adjoint_fd, args.output_dir
    )

    # Gradient accuracy assessment
    print("\nPerforming gradient accuracy assessment...")
    epsilon_range = 10.0**np.linspace(-6, 1, 64)
    eps_vec, dJ, dJ_predicted, u_adjoint_initial = compute_gradient_accuracy(
        validator, u_forward_fd, u_adjoint_fd, epsilon_range
    )
    plot_gradient_accuracy(eps_vec, dJ, dJ_predicted, u_adjoint_initial, args.output_dir)

    # Compute angle between NN and FD adjoint gradients
    nn_adj_gradient = adjoint_nn[0, :]
    fd_adj_gradient = u_adjoint_fd[:, 0]
    cosine_sim = np.inner(nn_adj_gradient, fd_adj_gradient) / \
                 np.sqrt(np.inner(nn_adj_gradient, nn_adj_gradient) *
                         np.inner(fd_adj_gradient, fd_adj_gradient))
    angle_deg = np.arccos(cosine_sim) * 180 / np.pi

    print(f"\nAdjoint gradient alignment:")
    print(f"  Cosine similarity: {cosine_sim:.4f}")
    print(f"  Angle: {angle_deg:.2f}°")

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
