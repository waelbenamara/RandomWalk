#!/usr/bin/env python3
"""
Create Registration Animation Script

This script runs the Gibbs sampler and saves frames every N iterations,
then creates an animated GIF showing how the point clouds converge.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from PIL import Image
import io

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_sampler import RealTimeGibbsSampler3D


class AnimatedGibbsSampler(RealTimeGibbsSampler3D):
    """
    Extended sampler that captures frames during iteration
    """
    
    def __init__(self, *args, frame_interval=2, max_frames=250, **kwargs):
        """
        Initialize animated sampler
        
        Args:
            frame_interval: Capture frame every N iterations (default: 2)
            max_frames: Maximum number of frames to capture (default: 250)
        """
        super().__init__(*args, **kwargs)
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.captured_frames = []
        self.frame_params = []
        
    def capture_frame(self, iteration, tx, ty, tz, roll, pitch, yaw):
        """Capture current state as a frame"""
        if len(self.captured_frames) >= self.max_frames:
            return
            
        self.frame_params.append({
            'iteration': iteration,
            'tx': tx, 'ty': ty, 'tz': tz,
            'roll': roll, 'pitch': pitch, 'yaw': yaw
        })
        
    def run_gibbs_sampler_animated(self, verbose=True):
        """Run Gibbs sampler with frame capture"""
        import time
        from sklearn.neighbors import NearestNeighbors
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"RUNNING ANIMATED GIBBS SAMPLER")
            print(f"{'='*70}")
            print(f"  Candidate points: {len(self.source_points)} source, {len(self.target_points)} target")
            print(f"  Total samples: {self.n_samples}")
            print(f"  Burn-in: {self.burnin}")
            print(f"  Frame interval: every {self.frame_interval} iterations")
            print(f"  Max frames: {self.max_frames}")
        
        total_start_time = time.time()
        
        # Initialize correspondences
        
        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nn.fit(self.target_points)
        _, initial_indices = nn.kneighbors(self.source_points)
        self.current_correspondences = initial_indices.flatten()
        
        # Pre-compute for faster sampling
        self.nn_sampler = NearestNeighbors(n_neighbors=min(20, len(self.target_points)), algorithm='kd_tree')
        self.nn_sampler.fit(self.target_points)
        
        # Capture initial frame
        self.capture_frame(0, self.current_tx, self.current_ty, self.current_tz,
                          self.current_roll, self.current_pitch, self.current_yaw)
        
        # Main Gibbs sampling loop
        for iteration in range(self.n_samples + self.burnin):
            # 1. Sample correspondences
            self.current_correspondences = self.sample_correspondences(
                self.current_tx, self.current_ty, self.current_tz,
                self.current_roll, self.current_pitch, self.current_yaw
            )
            
            # 2. Sample translation parameters
            self.current_tx = self.sample_tx(
                self.current_ty, self.current_tz, self.current_roll,
                self.current_pitch, self.current_yaw, self.current_correspondences
            )
            
            self.current_ty = self.sample_ty(
                self.current_tx, self.current_tz, self.current_roll,
                self.current_pitch, self.current_yaw, self.current_correspondences
            )
            
            self.current_tz = self.sample_tz(
                self.current_tx, self.current_ty, self.current_roll,
                self.current_pitch, self.current_yaw, self.current_correspondences
            )
            
            # 3. Sample rotation parameters
            self.current_roll = self.sample_rotation_parameter(
                'roll', self.current_tx, self.current_ty, self.current_tz,
                self.current_roll, self.current_pitch, self.current_yaw,
                self.current_correspondences
            )
            
            self.current_pitch = self.sample_rotation_parameter(
                'pitch', self.current_tx, self.current_ty, self.current_tz,
                self.current_roll, self.current_pitch, self.current_yaw,
                self.current_correspondences
            )
            
            self.current_yaw = self.sample_rotation_parameter(
                'yaw', self.current_tx, self.current_ty, self.current_tz,
                self.current_roll, self.current_pitch, self.current_yaw,
                self.current_correspondences
            )
            
            # Adapt proposal distributions
            self.adapt_proposal_stds(iteration)
            
            # Capture frame at specified interval
            if (iteration + 1) % self.frame_interval == 0:
                self.capture_frame(iteration + 1, 
                                 self.current_tx, self.current_ty, self.current_tz,
                                 self.current_roll, self.current_pitch, self.current_yaw)
                
                if verbose and (iteration + 1) % 50 == 0:
                    print(f"  Iteration {iteration + 1}/{self.n_samples + self.burnin} - Captured {len(self.frame_params)} frames")
            
            # Store samples (after burn-in)
            if iteration >= self.burnin and (iteration - self.burnin) % self.thin == 0:
                log_likelihood = self.compute_log_likelihood(
                    self.current_tx, self.current_ty, self.current_tz,
                    self.current_roll, self.current_pitch, self.current_yaw,
                    self.current_correspondences
                )
                
                self.samples['tx'].append(self.current_tx)
                self.samples['ty'].append(self.current_ty)
                self.samples['tz'].append(self.current_tz)
                self.samples['roll'].append(self.current_roll)
                self.samples['pitch'].append(self.current_pitch)
                self.samples['yaw'].append(self.current_yaw)
                self.samples['log_likelihood'].append(log_likelihood)
                self.samples['correspondences'].append(self.current_correspondences.copy())
        
        total_execution_time = time.time() - total_start_time
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Sampling complete!")
            print(f"  Total time: {total_execution_time:.3f}s")
            print(f"  Frames captured: {len(self.frame_params)}")
            print(f"{'='*70}")
        
        return True


def create_frame_image(sampler, params, frame_idx, total_frames, subsample_size=1000):
    """
    Create a single frame showing the current registration state
    
    Args:
        sampler: AnimatedGibbsSampler instance
        params: Dictionary with transformation parameters
        frame_idx: Current frame index
        total_frames: Total number of frames
        subsample_size: Number of points to visualize
    """
    # Subsample for visualization
    n_vis = min(subsample_size, len(sampler.source_points))
    source_indices = np.random.choice(len(sampler.source_points), n_vis, replace=False)
    target_indices = np.random.choice(len(sampler.target_points), n_vis, replace=False)
    
    source_vis = sampler.source_points[source_indices]
    target_vis = sampler.target_points[target_indices]
    
    # Apply current transformation
    transformed_source = sampler.apply_transformation(
        source_vis, params['tx'], params['ty'], params['tz'],
        params['roll'], params['pitch'], params['yaw']
    )
    
    # Compute error
    from scipy.spatial.distance import cdist
    distances = cdist(transformed_source, target_vis)
    min_distances = np.min(distances, axis=1)
    mean_error = np.mean(min_distances)
    
    # Compute parameter errors
    trans_error = np.sqrt(
        (params['tx'] - sampler.true_tx)**2 + 
        (params['ty'] - sampler.true_ty)**2 + 
        (params['tz'] - sampler.true_tz)**2
    )
    rot_error = np.degrees(np.sqrt(
        (params['roll'] - sampler.true_roll)**2 + 
        (params['pitch'] - sampler.true_pitch)**2 + 
        (params['yaw'] - sampler.true_yaw)**2
    ))
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot target (red)
    ax.scatter(target_vis[:, 0], target_vis[:, 1], target_vis[:, 2],
              c='crimson', alpha=0.6, s=20, label='Target',
              marker='o', edgecolors='darkred', linewidth=0.2)
    
    # Plot transformed source (blue)
    ax.scatter(transformed_source[:, 0], transformed_source[:, 1], transformed_source[:, 2],
              c='dodgerblue', alpha=0.6, s=20, label='Source (transformed)',
              marker='^', edgecolors='navy', linewidth=0.2)
    
    # Title with information
    iteration = params['iteration']
    title = (f'Gibbs Sampler Registration - Iteration {iteration}\n'
            f'Frame {frame_idx}/{total_frames} | '
            f'Mean Error: {mean_error:.5f}\n'
            f'Trans Error: {trans_error:.5f} | Rot Error: {rot_error:.3f}°')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper right')
    
    # Slow, cinematic rotating camera view
    # Complete only 0.5 rotations (180 degrees) over all frames
    progress = frame_idx / total_frames
    azim = progress * 180  # Half rotation for slower, more subtle movement
    
    # Add smooth elevation variation (wave up and down)
    # Elevation varies between 20° and 40° in a gentle wave
    elev = 30 + 10 * np.sin(progress * 2 * np.pi)  # Slower up-down motion
    
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3)
    
    # Convert to image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def create_animation_gif(sampler, output_path, subsample_size=1000, duration=50):
    """
    Create animated GIF from captured frames
    
    Args:
        sampler: AnimatedGibbsSampler instance with captured frames
        output_path: Path to save GIF
        subsample_size: Number of points to visualize per frame
        duration: Duration of each frame in milliseconds (default: 50ms for faster playback)
    """
    print(f"\n{'='*70}")
    print(f"CREATING ANIMATION GIF")
    print(f"{'='*70}")
    print(f"  Total frames: {len(sampler.frame_params)}")
    print(f"  Output: {output_path}")
    print(f"  Frame duration: {duration}ms")
    
    images = []
    total_frames = len(sampler.frame_params)
    
    for idx, params in enumerate(sampler.frame_params):
        print(f"  Rendering frame {idx+1}/{total_frames}...", end='\r')
        img = create_frame_image(sampler, params, idx+1, total_frames, subsample_size)
        images.append(img)
    
    print(f"\n  Saving GIF...")
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  GIF saved successfully!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"{'='*70}")


def main():
    """Main function to create registration animation"""
    print("="*70)
    print("GIBBS SAMPLER REGISTRATION ANIMATION")
    print("="*70)
    
    # Check if data exists
    data_dir = "/Users/waelbenamara/Desktop/Research/RandomWalk/data"
    point_cloud_file = f"{data_dir}/bunny_source.csv"
    
    if not os.path.exists(point_cloud_file):
        print(f"ERROR: Point cloud not found at {point_cloud_file}")
        return None
    
    # Define ground truth transformation
    true_tx, true_ty, true_tz = 0.9, 0.05, -0.08
    true_roll, true_pitch, true_yaw = 1.0, -0.1, 0.2
    
    print(f"\nGround Truth Transformation:")
    print(f"  Translation: ({true_tx:.4f}, {true_ty:.4f}, {true_tz:.4f})")
    print(f"  Rotation: ({np.degrees(true_roll):.2f}°, {np.degrees(true_pitch):.2f}°, {np.degrees(true_yaw):.2f}°)")
    
    # Create animated sampler
    print(f"\nInitializing animated sampler...")
    sampler = AnimatedGibbsSampler(
        point_cloud_file,
        true_tx=true_tx, true_ty=true_ty, true_tz=true_tz,
        true_roll=true_roll, true_pitch=true_pitch, true_yaw=true_yaw,
        candidate_ratio=0.1,
        extremity_ratio=0.7,
        add_noise=True,
        noise_std=0.001,
        frame_interval=2,      # Capture every 2 iterations
        max_frames=250         # Limit to 250 frames
    )
    
    # Run sampler with frame capture
    print(f"\nRunning Gibbs sampler with frame capture...")
    sampler.run_gibbs_sampler_animated(verbose=True)
    
    # Print final results
    print(f"\nFinal Results:")
    tx_est, ty_est, tz_est, roll_est, pitch_est, yaw_est = sampler.get_posterior_estimates()
    trans_error = np.sqrt((tx_est - true_tx)**2 + (ty_est - true_ty)**2 + (tz_est - true_tz)**2)
    rot_error = np.degrees(np.sqrt(
        (roll_est - true_roll)**2 + 
        (pitch_est - true_pitch)**2 + 
        (yaw_est - true_yaw)**2
    ))
    print(f"  Estimated: T=({tx_est:.4f}, {ty_est:.4f}, {tz_est:.4f})")
    print(f"             R=({np.degrees(roll_est):.2f}°, {np.degrees(pitch_est):.2f}°, {np.degrees(yaw_est):.2f}°)")
    print(f"  Translation error: {trans_error:.6f}")
    print(f"  Rotation error: {rot_error:.4f}°")
    
    # Create animation
    output_path = f"{data_dir}/registration_animation.gif"
    create_animation_gif(sampler, output_path, subsample_size=1000, duration=50)
    
    print(f"\nAnimation saved to: {output_path}")
    print(f"You can open it with any image viewer or web browser.")
    
    return sampler


if __name__ == "__main__":
    import time
    sampler = main()
    
    if sampler:
        print("\nSUCCESS: Animation created!")
    else:
        print("\nFAILED: Could not create animation")

