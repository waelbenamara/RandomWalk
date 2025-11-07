#!/usr/bin/env python3
"""
Performance comparison script for Cython optimizations

This script benchmarks the performance improvements achieved by Cython optimizations
in the 3D Gibbs sampler for point cloud registration.
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def generate_test_data(n_source=1000, n_target=1000):
    """Generate test point clouds for benchmarking"""
    np.random.seed(42)
    
    # Generate source points (sphere)
    phi = np.random.uniform(0, 2*np.pi, n_source)
    costheta = np.random.uniform(-1, 1, n_source)
    theta = np.arccos(costheta)
    
    radius = 2.0
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    source_points = np.column_stack([x, y, z]).astype(np.float64)
    
    # Generate target points (transformed sphere with noise)
    true_tx, true_ty, true_tz = 1.5, -0.8, 2.1
    true_roll, true_pitch, true_yaw = np.radians(15), np.radians(-10), np.radians(25)
    
    # Create rotation matrix
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(true_roll), -np.sin(true_roll)],
        [0, np.sin(true_roll), np.cos(true_roll)]
    ])
    
    R_y = np.array([
        [np.cos(true_pitch), 0, np.sin(true_pitch)],
        [0, 1, 0],
        [-np.sin(true_pitch), 0, np.cos(true_pitch)]
    ])
    
    R_z = np.array([
        [np.cos(true_yaw), -np.sin(true_yaw), 0],
        [np.sin(true_yaw), np.cos(true_yaw), 0],
        [0, 0, 1]
    ])
    
    R = R_z @ R_y @ R_x
    
    # Apply transformation and add noise
    target_points = (source_points @ R.T) + np.array([true_tx, true_ty, true_tz])
    target_points += np.random.normal(0, 0.05, target_points.shape)
    target_points = target_points.astype(np.float64)
    
    return source_points, target_points

def benchmark_transformation_functions():
    """Benchmark transformation and rotation matrix functions"""
    print("Benchmarking transformation functions...")
    
    # Test data
    n_points = 5000
    n_iterations = 100
    
    source_points, _ = generate_test_data(n_points, n_points)
    tx, ty, tz = 1.0, 2.0, 3.0
    roll, pitch, yaw = 0.1, 0.2, 0.3
    
    results = {}
    
    # Test original implementation
    print("  Testing original implementation...")
    from sampler_3d import GibbsSampler3DRegistration
    
    # Create a dummy sampler to access methods
    temp_source_file = "temp_source.csv"
    temp_target_file = "temp_target.csv"
    pd.DataFrame(source_points, columns=['x', 'y', 'z']).to_csv(temp_source_file, index=False)
    pd.DataFrame(source_points, columns=['x', 'y', 'z']).to_csv(temp_target_file, index=False)
    
    original_sampler = GibbsSampler3DRegistration(temp_source_file, temp_target_file)
    
    start_time = time.time()
    for _ in range(n_iterations):
        R = original_sampler.euler_to_rotation_matrix(roll, pitch, yaw)
        transformed = original_sampler.apply_transformation(source_points, tx, ty, tz, roll, pitch, yaw)
    original_time = time.time() - start_time
    results['Original'] = original_time
    
    # Clean up temp files
    import os
    os.remove(temp_source_file)
    os.remove(temp_target_file)
    
    # Test Cython implementation
    print("  Testing Cython implementation...")
    try:
        from cython_optimizations import euler_to_rotation_matrix_fast, apply_transformation_fast
        
        start_time = time.time()
        for _ in range(n_iterations):
            R = euler_to_rotation_matrix_fast(roll, pitch, yaw)
            transformed = apply_transformation_fast(source_points, tx, ty, tz, roll, pitch, yaw)
        cython_time = time.time() - start_time
        results['Cython'] = cython_time
        
        speedup = original_time / cython_time
        print(f"    Transformation speedup: {speedup:.2f}x")
        
    except ImportError:
        print("    Cython optimizations not available")
        results['Cython'] = None
    
    return results

def benchmark_likelihood_computation():
    """Benchmark likelihood computation functions"""
    print("Benchmarking likelihood computation...")
    
    # Test data
    n_points = 2000
    n_iterations = 50
    
    source_points, target_points = generate_test_data(n_points, n_points)
    correspondences = np.random.randint(0, n_points, n_points).astype(np.int32)
    tx, ty, tz = 1.0, 2.0, 3.0
    roll, pitch, yaw = 0.1, 0.2, 0.3
    noise_precision = 100.0
    
    results = {}
    
    # Test original implementation
    print("  Testing original implementation...")
    from sampler_3d import GibbsSampler3DRegistration
    
    temp_source_file = "temp_source.csv"
    temp_target_file = "temp_target.csv"
    pd.DataFrame(source_points, columns=['x', 'y', 'z']).to_csv(temp_source_file, index=False)
    pd.DataFrame(target_points, columns=['x', 'y', 'z']).to_csv(temp_target_file, index=False)
    
    original_sampler = GibbsSampler3DRegistration(temp_source_file, temp_target_file)
    
    start_time = time.time()
    for _ in range(n_iterations):
        log_lik = original_sampler.compute_log_likelihood(tx, ty, tz, roll, pitch, yaw, correspondences)
    original_time = time.time() - start_time
    results['Original'] = original_time
    
    # Clean up temp files
    import os
    os.remove(temp_source_file)
    os.remove(temp_target_file)
    
    # Test Cython implementation
    print("  Testing Cython implementation...")
    try:
        from cython_optimizations import compute_log_likelihood_fast
        
        start_time = time.time()
        for _ in range(n_iterations):
            log_lik = compute_log_likelihood_fast(
                source_points, target_points, correspondences,
                tx, ty, tz, roll, pitch, yaw, noise_precision
            )
        cython_time = time.time() - start_time
        results['Cython'] = cython_time
        
        speedup = original_time / cython_time
        print(f"    Likelihood computation speedup: {speedup:.2f}x")
        
    except ImportError:
        print("    Cython optimizations not available")
        results['Cython'] = None
    
    return results

def benchmark_full_sampler():
    """Benchmark the full sampler performance"""
    print("Benchmarking full sampler performance...")
    
    # Generate test data and save to files
    source_points, target_points = generate_test_data(1000, 1000)
    
    source_file = "temp_bench_source.csv"
    target_file = "temp_bench_target.csv"
    pd.DataFrame(source_points, columns=['x', 'y', 'z']).to_csv(source_file, index=False)
    pd.DataFrame(target_points, columns=['x', 'y', 'z']).to_csv(target_file, index=False)
    
    results = {}
    
    # Test original sampler
    print("  Testing original sampler (reduced iterations)...")
    from sampler_3d import GibbsSampler3DRegistration
    
    original_sampler = GibbsSampler3DRegistration(source_file, target_file)
    original_sampler.n_samples = 100  # Reduced for benchmarking
    original_sampler.burnin = 50
    
    start_time = time.time()
    original_sampler.run_gibbs_sampler(verbose=False)
    original_time = time.time() - start_time
    results['Original'] = original_time
    
    # Test optimized sampler
    print("  Testing optimized sampler...")
    try:
        from sampler_3d_optimized import OptimizedGibbsSampler3DRegistration
        
        optimized_sampler = OptimizedGibbsSampler3DRegistration(source_file, target_file)
        optimized_sampler.n_samples = 100  # Reduced for benchmarking
        optimized_sampler.burnin = 50
        
        start_time = time.time()
        optimized_sampler.run_gibbs_sampler(verbose=False)
        optimized_time = time.time() - start_time
        results['Optimized'] = optimized_time
        
        speedup = original_time / optimized_time
        print(f"    Full sampler speedup: {speedup:.2f}x")
        
    except ImportError:
        print("    Optimized sampler not available")
        results['Optimized'] = None
    
    # Clean up temp files
    import os
    os.remove(source_file)
    os.remove(target_file)
    
    return results

def plot_performance_results(transformation_results, likelihood_results, sampler_results):
    """Plot performance comparison results"""
    print("Generating performance plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Transformation benchmark
    if transformation_results['Cython'] is not None:
        speedup = transformation_results['Original'] / transformation_results['Cython']
        axes[0].bar(['Original', 'Cython'], 
                   [transformation_results['Original'], transformation_results['Cython']],
                   color=['lightcoral', 'lightblue'])
        axes[0].set_title(f'Transformation Functions\nSpeedup: {speedup:.2f}x')
        axes[0].set_ylabel('Time (seconds)')
        
        # Add speedup annotation
        axes[0].annotate(f'{speedup:.2f}x faster', 
                        xy=(1, transformation_results['Cython']), 
                        xytext=(1, transformation_results['Cython'] * 1.5),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        fontsize=12, ha='center', color='green', weight='bold')
    
    # Likelihood benchmark
    if likelihood_results['Cython'] is not None:
        speedup = likelihood_results['Original'] / likelihood_results['Cython']
        axes[1].bar(['Original', 'Cython'], 
                   [likelihood_results['Original'], likelihood_results['Cython']],
                   color=['lightcoral', 'lightblue'])
        axes[1].set_title(f'Likelihood Computation\nSpeedup: {speedup:.2f}x')
        axes[1].set_ylabel('Time (seconds)')
        
        # Add speedup annotation
        axes[1].annotate(f'{speedup:.2f}x faster', 
                        xy=(1, likelihood_results['Cython']), 
                        xytext=(1, likelihood_results['Cython'] * 1.5),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        fontsize=12, ha='center', color='green', weight='bold')
    
    # Full sampler benchmark
    if sampler_results['Optimized'] is not None:
        speedup = sampler_results['Original'] / sampler_results['Optimized']
        axes[2].bar(['Original', 'Optimized'], 
                   [sampler_results['Original'], sampler_results['Optimized']],
                   color=['lightcoral', 'lightgreen'])
        axes[2].set_title(f'Full Gibbs Sampler\nSpeedup: {speedup:.2f}x')
        axes[2].set_ylabel('Time (seconds)')
        
        # Add speedup annotation
        axes[2].annotate(f'{speedup:.2f}x faster', 
                        xy=(1, sampler_results['Optimized']), 
                        xytext=(1, sampler_results['Optimized'] * 1.5),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        fontsize=12, ha='center', color='green', weight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_benchmark_results(transformation_results, likelihood_results, sampler_results):
    """Save benchmark results to file"""
    print("Saving benchmark results...")
    
    with open('performance_benchmark_results.txt', 'w') as f:
        f.write("3D Gibbs Sampler Performance Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Transformation Functions:\n")
        f.write(f"  Original: {transformation_results['Original']:.4f} seconds\n")
        if transformation_results['Cython'] is not None:
            speedup = transformation_results['Original'] / transformation_results['Cython']
            f.write(f"  Cython: {transformation_results['Cython']:.4f} seconds\n")
            f.write(f"  Speedup: {speedup:.2f}x\n\n")
        else:
            f.write("  Cython: Not available\n\n")
        
        f.write("Likelihood Computation:\n")
        f.write(f"  Original: {likelihood_results['Original']:.4f} seconds\n")
        if likelihood_results['Cython'] is not None:
            speedup = likelihood_results['Original'] / likelihood_results['Cython']
            f.write(f"  Cython: {likelihood_results['Cython']:.4f} seconds\n")
            f.write(f"  Speedup: {speedup:.2f}x\n\n")
        else:
            f.write("  Cython: Not available\n\n")
        
        f.write("Full Gibbs Sampler:\n")
        f.write(f"  Original: {sampler_results['Original']:.4f} seconds\n")
        if sampler_results['Optimized'] is not None:
            speedup = sampler_results['Original'] / sampler_results['Optimized']
            f.write(f"  Optimized: {sampler_results['Optimized']:.4f} seconds\n")
            f.write(f"  Speedup: {speedup:.2f}x\n\n")
        else:
            f.write("  Optimized: Not available\n\n")

def main():
    """Main benchmarking process"""
    print("=" * 60)
    print("3D GIBBS SAMPLER PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Check if Cython optimizations are available
    try:
        import cython_optimizations
        print("✓ Cython optimizations available")
    except ImportError:
        print("! Cython optimizations not available")
        print("  Run 'python scripts/build_optimizations.py' to compile them")
    
    print("\nStarting performance benchmarks...\n")
    
    # Run benchmarks
    transformation_results = benchmark_transformation_functions()
    likelihood_results = benchmark_likelihood_computation()
    sampler_results = benchmark_full_sampler()
    
    # Generate plots and save results
    plot_performance_results(transformation_results, likelihood_results, sampler_results)
    save_benchmark_results(transformation_results, likelihood_results, sampler_results)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE!")
    print("=" * 60)
    print("Results saved to:")
    print("  • performance_comparison.png")
    print("  • performance_benchmark_results.txt")
    
    # Summary
    if transformation_results['Cython'] is not None:
        trans_speedup = transformation_results['Original'] / transformation_results['Cython']
        print(f"\nTransformation speedup: {trans_speedup:.2f}x")
    
    if likelihood_results['Cython'] is not None:
        lik_speedup = likelihood_results['Original'] / likelihood_results['Cython']
        print(f"Likelihood speedup: {lik_speedup:.2f}x")
    
    if sampler_results['Optimized'] is not None:
        sampler_speedup = sampler_results['Original'] / sampler_results['Optimized']
        print(f"Full sampler speedup: {sampler_speedup:.2f}x")

if __name__ == "__main__":
    main()
