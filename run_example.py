"""
Quick start example for AFDGM - runs complete workflow

This script demonstrates the full AFDGM pipeline:
1. Train forward and adjoint neural networks
2. Validate against finite difference solutions
3. Generate all plots and metrics

Author: David A. Buchta
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and print status"""
    print("\n" + "=" * 70)
    print(f"{description}")
    print("=" * 70)
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nError: {description} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n{description} completed successfully!")


def main():
    """Run complete AFDGM example"""

    print("=" * 70)
    print("AFDGM Complete Example")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Train forward and adjoint neural networks (quick demo: 500 epochs)")
    print("2. Validate solutions against finite difference")
    print("3. Generate all plots and metrics")
    print("\nEstimated time: 5-10 minutes")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")

    try:
        input()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)

    # Create output directory
    output_dir = "./output/"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Train the model (reduced epochs for quick demo)
    run_command(
        [sys.executable, "train_adjoint_advdiff.py",
         "--sampling_stages", "500",
         "--steps_per_sample", "10",
         "--output_dir", output_dir],
        "Step 1: Training Neural Networks"
    )

    # Step 2: Validate the solution
    run_command(
        [sys.executable, "validate_solution.py",
         "--output_dir", output_dir,
         "--nx", "100",
         "--nt", "200"],
        "Step 2: Validating Solutions"
    )

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - loss_evolution.png       : Training loss curves")
    print("  - solutions.png            : Forward and adjoint solutions")
    print("  - validation_comparison.png: NN vs FD comparison")
    print("  - gradient_accuracy.png    : Gradient verification")
    print("\nNext steps:")
    print("  - Examine plots in the output directory")
    print("  - For better accuracy, run with --sampling_stages 2000")
    print("  - Modify train_adjoint_advdiff.py to customize parameters")
    print("=" * 70)


if __name__ == "__main__":
    main()
