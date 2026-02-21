"""
Training script for Adjoint-Forward Deep Galerkin Method (AFDGM)
Applied to the advection-diffusion equation

Author: David A. Buchta
"""

import DGM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse

# Disable eager execution for TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()


class AdjointAdvDiffConfig:
    """Configuration parameters for the AFDGM solver"""

    # Space-time domain
    t_low = 0.0
    t_high = 1.0
    x_low = 0.0
    x_high = 1.0

    # Grid parameters
    n_interior = 100
    nx_interior = 100
    nt_interior = 100
    nSim_initial = 100
    nSim_terminal = 100
    nSim_boundaryA = 100
    nSim_boundaryB = 100

    # Neural network parameters
    num_layers = 3
    nodes_per_layer = 50
    applied_learning_rate = 1e-4

    # Training parameters
    sampling_stages = 2000
    steps_per_sample = 10

    # Plot parameters
    n_plot = 100

    # PDE parameters
    advection_speed = 1.0
    diffusion_coeff = 0.01

    # Output directory
    output_dir = './output/'


def reset_random_seeds(seed=1):
    """Reset all random seeds for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def initial_condition(nSim_initial, x_initial, config):
    """Gaussian initial condition centered at domain midpoint"""
    x_center = 0.5 * (config.x_high + config.x_low)
    uInitial = np.exp(-np.square(x_initial - x_center) / 0.01)
    return uInitial


def boundary_condition(nSim_boundaryA, nSim_boundaryB, x_a, t_a, x_b, t_b):
    """Zero boundary conditions at both boundaries"""
    uA = 0.0 * np.ones((nSim_boundaryA, 1))
    uB = 0.0 * np.ones((nSim_boundaryB, 1))
    return uA, uB


def grid(config):
    """Generate space-time sampling grid"""

    # Interior points (random sampling)
    t_interior = np.random.uniform(
        low=config.t_low, high=config.t_high, size=[config.nt_interior, 1]
    )
    x_interior = np.random.uniform(
        low=config.x_low, high=config.x_high, size=[config.nx_interior, 1]
    )

    # Initial condition points
    t_initial = config.t_low * np.ones((config.nSim_initial, 1))
    x_initial = np.linspace(config.x_low, config.x_high, config.nSim_initial)
    x_initial = x_initial.reshape((-1, 1))

    # Boundary A (left)
    x_a = config.x_low * np.ones((config.nSim_boundaryA, 1))
    t_a = np.linspace(config.t_low, config.t_high, config.nSim_boundaryA)
    t_a = t_a.reshape((-1, 1))

    # Boundary B (right)
    x_b = config.x_high * np.ones((config.nSim_boundaryB, 1))
    t_b = t_a

    # Terminal condition points
    t_terminal = config.t_high * np.ones((config.nSim_initial, 1))
    x_terminal = x_initial

    return (t_interior, x_interior, t_initial, x_initial,
            t_a, x_a, t_b, x_b, t_terminal, x_terminal)


def loss(model, adjmodel, t_interior, x_interior, t_initial, x_initial,
         t_a, x_a, t_b, x_b, t_terminal, x_terminal,
         uInitial, uA, uB, nu, c, parameters):
    """
    Compute total loss for forward and adjoint PDEs

    Returns:
        Ltotal: Total loss
        L1: Forward PDE residual
        L2: Forward IC loss
        L3: Forward BC loss
        adjL1: Adjoint PDE residual
        adjL2: Adjoint terminal condition loss
        adjL3: Adjoint BC loss
    """

    # Forward PDE: ∂u/∂t + c*∂u/∂x - ν*∂²u/∂x² = 0
    V = model(t_interior, x_interior, parameters)
    Vt = tf.gradients(ys=V, xs=t_interior)[0]
    Vx = tf.gradients(ys=V, xs=x_interior)[0]
    Vxx = tf.gradients(ys=Vx, xs=x_interior)[0]
    diff_V = Vt + c * Vx - nu * Vxx
    L1 = tf.reduce_mean(input_tensor=tf.square(diff_V))

    # Forward IC
    fitted_initial = model(t_initial, x_initial, parameters)
    L2 = tf.reduce_mean(input_tensor=tf.square(fitted_initial - uInitial))

    # Forward BC (periodic)
    fitted_a = model(t_a, x_a, parameters)
    Vxa = tf.gradients(ys=fitted_a, xs=x_a)[0]
    fitted_b = model(t_b, x_b, parameters)
    Vxb = tf.gradients(ys=fitted_b, xs=x_b)[0]
    L3 = tf.reduce_mean(input_tensor=tf.square(fitted_a - fitted_b))
    L3 = L3 + tf.reduce_mean(input_tensor=tf.square(Vxa - Vxb))

    # Adjoint PDE: ∂u†/∂t + c*∂u†/∂x + ν*∂u†/∂x = 0
    adjV = adjmodel(t_interior, x_interior, parameters)
    adjVt = tf.gradients(ys=adjV, xs=t_interior)[0]
    adjVx = tf.gradients(ys=adjV, xs=x_interior)[0]
    adjVxx = tf.gradients(ys=adjVx, xs=x_interior)[0]
    diff_adjV = adjVt + c * adjVx + nu * adjVx
    adjL1 = tf.reduce_mean(input_tensor=tf.square(diff_adjV))

    # Adjoint terminal condition: u†(T) = u(T)
    adj_fitted_terminal = adjmodel(t_terminal, x_terminal, parameters)
    fitted_terminal = model(t_terminal, x_terminal, parameters)
    adjL2 = tf.reduce_mean(input_tensor=tf.square(fitted_terminal - adj_fitted_terminal))

    # Adjoint BC (periodic)
    adj_fitted_a = adjmodel(t_a, x_a, parameters)
    adjVxa = tf.gradients(ys=adj_fitted_a, xs=x_a)[0]
    adj_fitted_b = adjmodel(t_b, x_b, parameters)
    adjVxb = tf.gradients(ys=adj_fitted_b, xs=x_b)[0]
    adjL3 = tf.reduce_mean(input_tensor=tf.square(adj_fitted_a - adj_fitted_b))
    adjL3 = adjL3 + tf.reduce_mean(input_tensor=tf.square(adjVxa - adjVxb))

    # Total loss
    Ltotal = L1 + L2 + L3 + adjL1 + adjL2 + adjL3

    return Ltotal, L1, L2, L3, adjL1, adjL2, adjL3


def setup_network(config):
    """Initialize neural networks and TensorFlow graph"""

    # Initialize DGM models
    model = DGM.DGMNet(config.nodes_per_layer, config.num_layers, 1 + config.nSim_initial)
    adjmodel = DGM.DGMNet(config.nodes_per_layer, config.num_layers, 1 + config.nSim_initial)

    # Tensor placeholders
    t_interior_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    x_interior_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    t_initial_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    x_initial_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    t_terminal_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    x_terminal_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    t_a_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    x_a_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    t_b_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    x_b_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    uInitial_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    uA_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    uB_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
    learning_rate = tf.compat.v1.placeholder(tf.float32)
    nu_tnsr = tf.compat.v1.placeholder(tf.float32)
    c_tnsr = tf.compat.v1.placeholder(tf.float32)
    parameters_tnsr = tf.compat.v1.placeholder(tf.float32, [None, None])

    # Loss computation
    loss_tnsr, L1_tnsr, L2_tnsr, L3_tnsr, adjL1_tnsr, adjL2_tnsr, adjL3_tnsr = loss(
        model, adjmodel,
        t_interior_tnsr, x_interior_tnsr,
        t_initial_tnsr, x_initial_tnsr,
        t_a_tnsr, x_a_tnsr, t_b_tnsr, x_b_tnsr,
        t_terminal_tnsr, x_terminal_tnsr,
        uInitial_tnsr, uA_tnsr, uB_tnsr, nu_tnsr, c_tnsr, parameters_tnsr
    )

    # Model outputs for inference
    V = model(t_interior_tnsr, x_interior_tnsr, parameters_tnsr)
    adjV = adjmodel(t_interior_tnsr, x_interior_tnsr, parameters_tnsr)

    # Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

    # Initialize session
    init_op = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init_op)

    placeholders = {
        't_interior': t_interior_tnsr,
        'x_interior': x_interior_tnsr,
        't_initial': t_initial_tnsr,
        'x_initial': x_initial_tnsr,
        't_terminal': t_terminal_tnsr,
        'x_terminal': x_terminal_tnsr,
        't_a': t_a_tnsr,
        'x_a': x_a_tnsr,
        't_b': t_b_tnsr,
        'x_b': x_b_tnsr,
        'uInitial': uInitial_tnsr,
        'uA': uA_tnsr,
        'uB': uB_tnsr,
        'learning_rate': learning_rate,
        'nu': nu_tnsr,
        'c': c_tnsr,
        'parameters': parameters_tnsr
    }

    tensors = {
        'loss': loss_tnsr,
        'L1': L1_tnsr,
        'L2': L2_tnsr,
        'L3': L3_tnsr,
        'adjL1': adjL1_tnsr,
        'adjL2': adjL2_tnsr,
        'adjL3': adjL3_tnsr,
        'optimizer': optimizer,
        'V': V,
        'adjV': adjV
    }

    return sess, model, adjmodel, placeholders, tensors


def train(config, verbose=True):
    """Train the forward and adjoint neural networks"""

    # Reset seeds
    reset_random_seeds()

    # Setup network
    sess, model, adjmodel, placeholders, tensors = setup_network(config)

    # Training loop
    lossData = np.zeros((config.sampling_stages, 8))
    current_lr = config.applied_learning_rate

    print("Starting training...")
    print(f"Sampling stages: {config.sampling_stages}")
    print(f"Steps per sample: {config.steps_per_sample}")

    for i in range(config.sampling_stages):

        # Generate new sampling grid
        (t_interior, x_interior, t_initial, x_initial,
         t_a, x_a, t_b, x_b, t_terminal, x_terminal) = grid(config)

        # Initial and boundary conditions
        uInitial = initial_condition(config.nSim_initial, x_initial, config)
        uA, uB = boundary_condition(config.nSim_boundaryA, config.nSim_boundaryB,
                                     x_a, t_a, x_b, t_b)

        # Parameters (initial condition perturbations)
        parameters = np.zeros((config.n_interior, config.nSim_initial))
        for n in range(config.n_interior):
            parameters[n, :] = np.transpose(uInitial)

        # Learning rate decay
        if (i + 1) % 500 == 0:
            current_lr = current_lr / (10**0.5)
            if verbose:
                print(f"Epoch {i+1}: Reducing learning rate to {current_lr:.2e}")

        # SGD steps
        for j in range(config.steps_per_sample):
            loss_val, L1, L2, L3, adjL1, adjL2, adjL3, _ = sess.run(
                [tensors['loss'], tensors['L1'], tensors['L2'], tensors['L3'],
                 tensors['adjL1'], tensors['adjL2'], tensors['adjL3'], tensors['optimizer']],
                feed_dict={
                    placeholders['t_interior']: t_interior,
                    placeholders['x_interior']: x_interior,
                    placeholders['t_initial']: t_initial,
                    placeholders['x_initial']: x_initial,
                    placeholders['t_a']: t_a,
                    placeholders['x_a']: x_a,
                    placeholders['t_b']: t_b,
                    placeholders['x_b']: x_b,
                    placeholders['uInitial']: uInitial,
                    placeholders['uA']: uA,
                    placeholders['uB']: uB,
                    placeholders['t_terminal']: t_terminal,
                    placeholders['x_terminal']: x_terminal,
                    placeholders['learning_rate']: current_lr,
                    placeholders['nu']: config.diffusion_coeff,
                    placeholders['c']: config.advection_speed,
                    placeholders['parameters']: parameters
                }
            )

        # Store loss data
        lossData[i, 0] = i
        lossData[i, 1] = loss_val
        lossData[i, 2] = L1
        lossData[i, 3] = L2
        lossData[i, 4] = L3
        lossData[i, 5] = adjL1
        lossData[i, 6] = adjL2
        lossData[i, 7] = adjL3

        if verbose and (i % 100 == 0 or i == config.sampling_stages - 1):
            print(f"Epoch {i:4d}: Loss={loss_val:.2e}, L_PDE={L1:.2e}, "
                  f"L_IC={L2:.2e}, L_BC={L3:.2e}, "
                  f"L†_PDE={adjL1:.2e}, L†_TC={adjL2:.2e}, L†_BC={adjL3:.2e}")

    return sess, model, adjmodel, placeholders, tensors, lossData, uInitial


def save_results(sess, model, adjmodel, placeholders, tensors, lossData, uInitial, config):
    """Save training results and generate plots"""

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Save loss data
    np.save(os.path.join(config.output_dir, 'loss_data.npy'), lossData)

    # Generate solution mesh
    nx = config.n_plot
    nt = config.n_plot
    x_plot = np.linspace(config.x_low, config.x_high, nx)
    t_plot = np.linspace(config.t_low, config.t_high, nt)
    x_mesh, t_mesh = np.meshgrid(x_plot, t_plot)

    t_plot_vec = np.reshape(t_mesh, [nx * nt, 1])
    x_plot_vec = np.reshape(x_mesh, [nx * nt, 1])

    # Parameters
    parameters = np.zeros((nx * nt, config.nSim_initial))
    for n in range(nx * nt):
        parameters[n, :] = np.transpose(uInitial)

    # Evaluate forward and adjoint solutions
    forwdValue = sess.run([tensors['V']], feed_dict={
        placeholders['t_interior']: t_plot_vec,
        placeholders['x_interior']: x_plot_vec,
        placeholders['parameters']: parameters
    })
    forwdValue_mesh = np.reshape(forwdValue, [nt, nx])

    adjValue = sess.run([tensors['adjV']], feed_dict={
        placeholders['t_interior']: t_plot_vec,
        placeholders['x_interior']: x_plot_vec,
        placeholders['parameters']: parameters
    })
    adjValue_mesh = np.reshape(adjValue, [nt, nx])

    # Save solutions
    np.save(os.path.join(config.output_dir, 'forward_solution.npy'), forwdValue_mesh)
    np.save(os.path.join(config.output_dir, 'adjoint_solution.npy'), adjValue_mesh)
    np.save(os.path.join(config.output_dir, 'x_mesh.npy'), x_mesh)
    np.save(os.path.join(config.output_dir, 't_mesh.npy'), t_mesh)

    # Plot loss evolution
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.semilogy(lossData[:, 0], lossData[:, 1], label='Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(132)
    plt.semilogy(lossData[:, 0], lossData[:, 2], label='L_PDE')
    plt.semilogy(lossData[:, 0], lossData[:, 3], label='L_IC')
    plt.semilogy(lossData[:, 0], lossData[:, 4], label='L_BC')
    plt.xlabel('Epochs')
    plt.ylabel('Forward Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(133)
    plt.semilogy(lossData[:, 0], lossData[:, 5], label='L†_PDE')
    plt.semilogy(lossData[:, 0], lossData[:, 6], label='L†_TC')
    plt.semilogy(lossData[:, 0], lossData[:, 7], label='L†_BC')
    plt.xlabel('Epochs')
    plt.ylabel('Adjoint Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'loss_evolution.png'), dpi=150)
    print(f"Saved loss plot to {config.output_dir}loss_evolution.png")

    # Plot solutions
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.pcolormesh(x_mesh, t_mesh, forwdValue_mesh, cmap="turbo", shading='gouraud')
    plt.colorbar(label='u')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Forward Solution u(x,t)')

    plt.subplot(122)
    plt.pcolormesh(x_mesh, t_mesh, adjValue_mesh, cmap="turbo", shading='gouraud')
    plt.colorbar(label='u†')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Adjoint Solution u†(x,t)')

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'solutions.png'), dpi=150)
    print(f"Saved solutions plot to {config.output_dir}solutions.png")

    return forwdValue_mesh, adjValue_mesh, x_mesh, t_mesh


def main():
    """Main training script"""

    parser = argparse.ArgumentParser(description='Train AFDGM for advection-diffusion equation')
    parser.add_argument('--sampling_stages', type=int, default=2000,
                        help='Number of sampling stages (default: 2000)')
    parser.add_argument('--steps_per_sample', type=int, default=10,
                        help='SGD steps per sample (default: 10)')
    parser.add_argument('--output_dir', type=str, default='./output/',
                        help='Output directory (default: ./output/)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')

    args = parser.parse_args()

    # Create config
    config = AdjointAdvDiffConfig()
    config.sampling_stages = args.sampling_stages
    config.steps_per_sample = args.steps_per_sample
    config.output_dir = args.output_dir
    config.applied_learning_rate = args.learning_rate

    print("=" * 70)
    print("Adjoint-Forward Deep Galerkin Method (AFDGM)")
    print("Advection-Diffusion Equation")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Domain: x ∈ [{config.x_low}, {config.x_high}], t ∈ [{config.t_low}, {config.t_high}]")
    print(f"  Advection speed: c = {config.advection_speed}")
    print(f"  Diffusion coeff: ν = {config.diffusion_coeff}")
    print(f"  Network: {config.num_layers} layers, {config.nodes_per_layer} nodes/layer")
    print(f"  Training: {config.sampling_stages} epochs, {config.steps_per_sample} steps/epoch")
    print("=" * 70)

    # Train
    sess, model, adjmodel, placeholders, tensors, lossData, uInitial = train(config)

    # Save results
    print("\nSaving results...")
    forwdValue_mesh, adjValue_mesh, x_mesh, t_mesh = save_results(
        sess, model, adjmodel, placeholders, tensors, lossData, uInitial, config
    )

    print("\nTraining complete!")
    print(f"Final loss: {lossData[-1, 1]:.2e}")
    print(f"Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()
