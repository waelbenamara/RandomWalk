import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
import json
import os
from scipy.stats import norm, multivariate_normal, uniform
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from collections import defaultdict

class RealTimeGibbsSampler3D:
    """
    Real-Time Optimized Gibbs Sampler for Bayesian 3D Point Cloud Registration
    
    Key Optimizations:
    1. Strategic candidate point selection (extremities + centroids)
    2. Reduced computational complexity
    3. Adaptive sampling for real-time performance
    4. Comprehensive timing measurements
    """
    
    def __init__(self, point_cloud_file, true_tx=0.1, true_ty=0.05, true_tz=-0.08, 
                 true_roll=0.15, true_pitch=-0.1, true_yaw=0.2, 
                 candidate_ratio=0.05, extremity_ratio=0.7, add_noise=True, noise_std=0.001):
        """
        Initialize real-time Gibbs sampler with single point cloud
        
        Args:
            point_cloud_file: Path to point cloud CSV (will be used as source)
            true_tx, true_ty, true_tz: Ground truth translation parameters
            true_roll, true_pitch, true_yaw: Ground truth rotation parameters (in radians)
            candidate_ratio: Fraction of points to use (default: 0.05 = 5%)
            extremity_ratio: Fraction of candidates from extremities (default: 0.7 = 70%)
            add_noise: Whether to add noise to the target point cloud
            noise_std: Standard deviation of Gaussian noise to add
        """
        self.timing_stats = defaultdict(float)
        self.timing_counts = defaultdict(int)
        
        # Store ground truth transformation
        self.true_tx = true_tx
        self.true_ty = true_ty
        self.true_tz = true_tz
        self.true_roll = true_roll
        self.true_pitch = true_pitch
        self.true_yaw = true_yaw
        
        # Load source point cloud
        t_start = time.time()
        self.full_source_points = pd.read_csv(point_cloud_file).values
        
        # Ensure 3D points
        if self.full_source_points.shape[1] != 3:
            raise ValueError("Input point cloud must be 3D (have 3 columns: x, y, z)")
        
        # Apply ground truth transformation to create target
        self.full_target_points = self.apply_transformation(
            self.full_source_points, true_tx, true_ty, true_tz, true_roll, true_pitch, true_yaw
        )
        
        # Add noise if requested
        if add_noise:
            noise = np.random.normal(0, noise_std, self.full_target_points.shape)
            self.full_target_points = self.full_target_points + noise
            print(f"Added Gaussian noise with σ={noise_std}")
        
        self.timing_stats['data_loading'] = time.time() - t_start
        
        print(f"Loaded {len(self.full_source_points)} points from source")
        print(f"Created target with transformation: T=({true_tx:.4f}, {true_ty:.4f}, {true_tz:.4f}), " +
              f"R=({np.degrees(true_roll):.2f}°, {np.degrees(true_pitch):.2f}°, {np.degrees(true_yaw):.2f}°)")
        
        # Candidate selection parameters
        self.candidate_ratio = candidate_ratio
        self.extremity_ratio = extremity_ratio
        
        # Select candidate points strategically
        t_start = time.time()
        self.select_candidate_points()
        self.timing_stats['candidate_selection'] = time.time() - t_start
        
        print(f"\nCandidate Point Selection:")
        print(f"  Candidate ratio: {self.candidate_ratio*100:.1f}%")
        print(f"  Extremity ratio: {self.extremity_ratio*100:.1f}%")
        print(f"  Source candidates: {len(self.source_points)} ({len(self.source_points)/len(self.full_source_points)*100:.1f}%)")
        print(f"  Target candidates: {len(self.target_points)} ({len(self.target_points)/len(self.full_target_points)*100:.1f}%)")
        print(f"  Selection time: {self.timing_stats['candidate_selection']:.3f}s")
        
        # Model hyperparameters
        self.setup_priors()
        
        # MCMC parameters (optimized for real-time)
        self.n_samples = 2000   # Reduced for real-time
        self.burnin = 5       # Reduced burn-in
        self.thin = 1
        
        # Current state - 3D transformation parameters
        self.current_tx = 0.0
        self.current_ty = 0.0
        self.current_tz = 0.0
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_correspondences = np.zeros(len(self.source_points), dtype=int)
        
        # Storage for samples
        self.samples = {
            'tx': [], 'ty': [], 'tz': [],
            'roll': [], 'pitch': [], 'yaw': [],
            'log_likelihood': [],
            'correspondences': []
        }
        
        # Adaptive MCMC parameters
        self.adaptive_mcmc = True
        self.adaptation_interval = 25
        self.target_acceptance_rate = 0.44
        self.adaptation_factor = 1.01
        
        # Proposal standard deviations
        self.proposal_stds = {
            'roll': 0.05, 'pitch': 0.05, 'yaw': 0.1
        }
        
        # Track acceptance rates
        self.recent_acceptances = {
            'roll': [], 'pitch': [], 'yaw': []
        }
        
    def select_candidate_points(self):
        """
        Strategically select candidate points from extremities and centroids
        
        Strategy:
        1. Extremity points: Points far from centroid (boundary/edge points)
        2. Centroid-proximal points: Points near centroid for translation estimation
        3. Random sampling for diversity
        """
        # Calculate number of candidates
        n_source_candidates = max(50, int(len(self.full_source_points) * self.candidate_ratio))
        n_target_candidates = max(50, int(len(self.full_target_points) * self.candidate_ratio))
        
        # Select source candidates
        self.source_points, self.source_indices = self._select_strategic_points(
            self.full_source_points, n_source_candidates
        )
        
        # Select target candidates
        self.target_points, self.target_indices = self._select_strategic_points(
            self.full_target_points, n_target_candidates
        )
        
    def _select_strategic_points(self, points, n_candidates):
        """
        Improved strategic point selection using multiple criteria
        """
        # Calculate different selection ratios
        n_extremity = int(n_candidates * self.extremity_ratio)
        n_surface = int(n_candidates * 0.15)  # 15% surface/edge points
        n_uniform = int(n_candidates * 0.10)  # 10% uniform sampling
        n_random = n_candidates - n_extremity - n_surface - n_uniform
        
        # Compute centroid and principal axes
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # PCA for better geometric understanding
        cov_matrix = np.cov(centered_points.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # 1. Extremity points (furthest from centroid)
        distances_from_center = np.linalg.norm(centered_points, axis=1)
        extremity_indices = np.argsort(distances_from_center)[-n_extremity:]
        
        # 2. Surface/edge points (high local variance)
        surface_indices = self._select_surface_points(points, n_surface)
        
        # 3. Uniform grid sampling
        uniform_indices = self._select_uniform_grid_points(points, n_uniform)
        
        # 4. Random selection from remaining points
        used_indices = np.concatenate([extremity_indices, surface_indices, uniform_indices])
        used_indices = np.unique(used_indices)
        remaining_indices = np.setdiff1d(np.arange(len(points)), used_indices)
        
        if len(remaining_indices) >= n_random and n_random > 0:
            random_indices = np.random.choice(remaining_indices, n_random, replace=False)
        else:
            random_indices = np.array([], dtype=int)
        
        # Combine all selected indices
        selected_indices = np.concatenate([extremity_indices, surface_indices, uniform_indices, random_indices])
        selected_indices = np.unique(selected_indices)
        
        # Ensure we have the right number of candidates
        if len(selected_indices) > n_candidates:
            # Prioritize extremity points, then others
            priority_order = np.concatenate([extremity_indices, surface_indices, uniform_indices, random_indices])
            selected_indices = priority_order[:n_candidates]
        elif len(selected_indices) < n_candidates:
            # Fill remaining with random points
            all_remaining = np.setdiff1d(np.arange(len(points)), selected_indices)
            if len(all_remaining) > 0:
                additional_needed = n_candidates - len(selected_indices)
                additional = np.random.choice(all_remaining, 
                                            min(additional_needed, len(all_remaining)), 
                                            replace=False)
                selected_indices = np.concatenate([selected_indices, additional])
        
        return points[selected_indices], selected_indices
    
    def _select_surface_points(self, points, n_surface):
        """Select points that are likely on the surface/edges"""
        if n_surface == 0:
            return np.array([], dtype=int)
            
        # Use local neighborhood variance as surface indicator
        nn = NearestNeighbors(n_neighbors=min(10, len(points)), algorithm='kd_tree')
        nn.fit(points)
        
        # Sample subset for efficiency
        sample_size = min(2000, len(points))
        sample_indices = np.random.choice(len(points), sample_size, replace=False)
        sample_points = points[sample_indices]
        
        distances, neighbor_indices = nn.kneighbors(sample_points)
        
        # Compute local variance (surface points have higher variance)
        local_variances = []
        for i, neighbors in enumerate(neighbor_indices):
            neighbor_points = points[neighbors]
            local_var = np.var(neighbor_points, axis=0).sum()
            local_variances.append(local_var)
        
        # Select points with highest local variance
        surface_candidates = sample_indices[np.argsort(local_variances)[-n_surface:]]
        return surface_candidates
    
    def _select_uniform_grid_points(self, points, n_uniform):
        """Select points using uniform grid sampling"""
        if n_uniform == 0:
            return np.array([], dtype=int)
            
        # Create 3D grid
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Determine grid resolution
        grid_size = int(np.ceil(n_uniform ** (1/3)))  # Cube root for 3D
        
        # Create grid
        x_bins = np.linspace(min_coords[0], max_coords[0], grid_size + 1)
        y_bins = np.linspace(min_coords[1], max_coords[1], grid_size + 1)
        z_bins = np.linspace(min_coords[2], max_coords[2], grid_size + 1)
        
        # Assign points to grid cells and sample one per cell
        selected_indices = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(selected_indices) >= n_uniform:
                        break
                        
                    # Find points in this grid cell
                    in_cell = ((points[:, 0] >= x_bins[i]) & (points[:, 0] < x_bins[i+1]) &
                              (points[:, 1] >= y_bins[j]) & (points[:, 1] < y_bins[j+1]) &
                              (points[:, 2] >= z_bins[k]) & (points[:, 2] < z_bins[k+1]))
                    
                    cell_indices = np.where(in_cell)[0]
                    if len(cell_indices) > 0:
                        # Select random point from this cell
                        selected_indices.append(np.random.choice(cell_indices))
                        
                if len(selected_indices) >= n_uniform:
                    break
            if len(selected_indices) >= n_uniform:
                break
        
        return np.array(selected_indices[:n_uniform])
    
    def setup_priors(self):
        """Setup adaptive prior distributions for 3D"""
        # Analyze candidate data to set reasonable priors
        source_center = np.mean(self.source_points, axis=0)
        target_center = np.mean(self.target_points, axis=0)
        center_diff = target_center - source_center
        
        # Compute data scale
        source_scale = np.std(self.source_points, axis=0)
        target_scale = np.std(self.target_points, axis=0)
        data_scale = np.mean([source_scale, target_scale], axis=0)
        overall_scale = np.mean(data_scale)
        
        # Translation priors
        self.prior_tx_mean = center_diff[0]
        self.prior_tx_var = (3 * overall_scale)**2
        
        self.prior_ty_mean = center_diff[1]
        self.prior_ty_var = (3 * overall_scale)**2
        
        self.prior_tz_mean = center_diff[2]
        self.prior_tz_var = (3 * overall_scale)**2
        
        # Rotation priors
        self.prior_roll_mean = 0.0
        self.prior_roll_var = (np.pi/3)**2
        
        self.prior_pitch_mean = 0.0
        self.prior_pitch_var = (np.pi/3)**2
        
        self.prior_yaw_mean = 0.0
        self.prior_yaw_var = (np.pi/3)**2
        
        # Adaptive noise parameter
        nn_distances = self.compute_nearest_neighbor_distances()
        median_nn_distance = np.median(nn_distances)
        self.noise_precision = 1.0 / (median_nn_distance**2)
        
        print(f"\n3D Prior setup (on candidate points):")
        print(f"  TX prior: N({self.prior_tx_mean:.2f}, {np.sqrt(self.prior_tx_var):.2f})")
        print(f"  TY prior: N({self.prior_ty_mean:.2f}, {np.sqrt(self.prior_ty_var):.2f})")
        print(f"  TZ prior: N({self.prior_tz_mean:.2f}, {np.sqrt(self.prior_tz_var):.2f})")
        print(f"  Noise precision: {self.noise_precision:.2f} (σ = {1/np.sqrt(self.noise_precision):.3f})")
        print(f"  Data scale: {overall_scale:.4f}, NN distance: {median_nn_distance:.4f}")
        
    def compute_nearest_neighbor_distances(self):
        """Compute nearest neighbor distances for adaptive noise estimation"""
        n_sample = min(500, len(self.source_points))
        indices = np.random.choice(len(self.source_points), n_sample, replace=False)
        sample_points = self.source_points[indices]
        
        nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
        nn.fit(sample_points)
        distances, _ = nn.kneighbors(sample_points)
        return distances[:, 1]
    
    def adapt_proposal_stds(self, iteration):
        """Adapt proposal standard deviations"""
        if not self.adaptive_mcmc or iteration < self.adaptation_interval:
            return
            
        if iteration % self.adaptation_interval == 0:
            for param in ['roll', 'pitch', 'yaw']:
                if len(self.recent_acceptances[param]) > 0:
                    acceptance_rate = np.mean(self.recent_acceptances[param])
                    
                    if acceptance_rate > self.target_acceptance_rate:
                        self.proposal_stds[param] *= self.adaptation_factor
                    else:
                        self.proposal_stds[param] /= self.adaptation_factor
                    
                    self.proposal_stds[param] = np.clip(self.proposal_stds[param], 0.001, 0.5)
                    self.recent_acceptances[param] = []
    
    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Convert Euler angles to 3D rotation matrix (ZYX convention)"""
        # Roll (rotation around x-axis)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch (rotation around y-axis)
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw (rotation around z-axis)
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix (ZYX order)
        R = R_z @ R_y @ R_x
        return R
    
    def apply_transformation(self, points, tx, ty, tz, roll, pitch, yaw):
        """Apply 3D rigid transformation"""
        rotation_matrix = self.euler_to_rotation_matrix(roll, pitch, yaw)
        rotated_points = points @ rotation_matrix.T
        transformed_points = rotated_points + np.array([tx, ty, tz])
        return transformed_points
    
    def compute_log_likelihood(self, tx, ty, tz, roll, pitch, yaw, correspondences):
        """Compute log likelihood of current state"""
        t_start = time.time()
        
        transformed_source = self.apply_transformation(
            self.source_points, tx, ty, tz, roll, pitch, yaw
        )
        
        target_correspondences = self.target_points[correspondences]
        squared_distances = np.sum((transformed_source - target_correspondences)**2, axis=1)
        
        log_likelihood = -0.5 * self.noise_precision * np.sum(squared_distances)
        log_likelihood -= 0.5 * len(self.source_points) * np.log(2 * np.pi / self.noise_precision)
        
        self.timing_stats['log_likelihood'] += time.time() - t_start
        self.timing_counts['log_likelihood'] += 1
        
        return log_likelihood
    
    def sample_correspondences(self, tx, ty, tz, roll, pitch, yaw):
        """Sample correspondence variables (optimized)"""
        t_start = time.time()
        
        transformed_source = self.apply_transformation(
            self.source_points, tx, ty, tz, roll, pitch, yaw
        )
        
        # Use k nearest neighbors for efficiency
        k_neighbors = min(20, len(self.target_points))  # Reduced k for speed
        distances, neighbor_indices = self.nn_sampler.kneighbors(transformed_source)
        
        distances_squared = distances ** 2
        log_probs = -0.5 * self.noise_precision * distances_squared
        
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - max_log_probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        
        cumprobs = np.cumsum(probs, axis=1)
        random_vals = np.random.random(len(transformed_source))[:, np.newaxis]
        local_indices = np.argmax(cumprobs >= random_vals, axis=1)
        
        new_correspondences = neighbor_indices[np.arange(len(transformed_source)), local_indices]
        
        self.timing_stats['sample_correspondences'] += time.time() - t_start
        self.timing_counts['sample_correspondences'] += 1
        
        return new_correspondences
    
    def sample_tx(self, ty, tz, roll, pitch, yaw, correspondences):
        """Sample tx given other parameters"""
        t_start = time.time()
        
        transformed_without_tx = self.apply_transformation(
            self.source_points, 0, ty, tz, roll, pitch, yaw
        )
        target_correspondences = self.target_points[correspondences]
        
        n = len(self.source_points)
        sum_diff_x = np.sum(target_correspondences[:, 0] - transformed_without_tx[:, 0])
        
        posterior_precision = self.noise_precision * n + 1.0 / self.prior_tx_var
        posterior_mean = (self.noise_precision * sum_diff_x + self.prior_tx_mean / self.prior_tx_var) / posterior_precision
        posterior_var = 1.0 / posterior_precision
        
        result = np.random.normal(posterior_mean, np.sqrt(posterior_var))
        
        self.timing_stats['sample_translation'] += time.time() - t_start
        self.timing_counts['sample_translation'] += 1
        
        return result
    
    def sample_ty(self, tx, tz, roll, pitch, yaw, correspondences):
        """Sample ty given other parameters"""
        transformed_without_ty = self.apply_transformation(
            self.source_points, tx, 0, tz, roll, pitch, yaw
        )
        target_correspondences = self.target_points[correspondences]
        
        n = len(self.source_points)
        sum_diff_y = np.sum(target_correspondences[:, 1] - transformed_without_ty[:, 1])
        
        posterior_precision = self.noise_precision * n + 1.0 / self.prior_ty_var
        posterior_mean = (self.noise_precision * sum_diff_y + self.prior_ty_mean / self.prior_ty_var) / posterior_precision
        posterior_var = 1.0 / posterior_precision
        
        return np.random.normal(posterior_mean, np.sqrt(posterior_var))
    
    def sample_tz(self, tx, ty, roll, pitch, yaw, correspondences):
        """Sample tz given other parameters"""
        transformed_without_tz = self.apply_transformation(
            self.source_points, tx, ty, 0, roll, pitch, yaw
        )
        target_correspondences = self.target_points[correspondences]
        
        n = len(self.source_points)
        sum_diff_z = np.sum(target_correspondences[:, 2] - transformed_without_tz[:, 2])
        
        posterior_precision = self.noise_precision * n + 1.0 / self.prior_tz_var
        posterior_mean = (self.noise_precision * sum_diff_z + self.prior_tz_mean / self.prior_tz_var) / posterior_precision
        posterior_var = 1.0 / posterior_precision
        
        return np.random.normal(posterior_mean, np.sqrt(posterior_var))
    
    def sample_rotation_parameter(self, param_name, tx, ty, tz, roll, pitch, yaw, correspondences):
        """Sample rotation parameter using Metropolis-Hastings"""
        t_start = time.time()
        
        param_map = {
            'roll': (roll, self.prior_roll_mean, self.prior_roll_var, 0),
            'pitch': (pitch, self.prior_pitch_mean, self.prior_pitch_var, 1),
            'yaw': (yaw, self.prior_yaw_mean, self.prior_yaw_var, 2)
        }
        
        current_val, prior_mean, prior_var, param_idx = param_map[param_name]
        
        proposal_std = self.proposal_stds[param_name]
        proposed_val = current_val + np.random.normal(0, proposal_std)
        
        current_params = np.array([roll, pitch, yaw])
        proposed_params = current_params.copy()
        proposed_params[param_idx] = proposed_val
        
        if abs(proposed_val - current_val) < 1e-6:
            self.timing_stats['sample_rotation'] += time.time() - t_start
            self.timing_counts['sample_rotation'] += 1
            return current_val
            
        log_likelihood_current = self.compute_log_likelihood(tx, ty, tz, *current_params, correspondences)
        log_likelihood_proposed = self.compute_log_likelihood(tx, ty, tz, *proposed_params, correspondences)
        
        log_prior_current = -0.5 * (current_val - prior_mean)**2 / prior_var
        log_prior_proposed = -0.5 * (proposed_val - prior_mean)**2 / prior_var
        
        log_acceptance = (log_likelihood_proposed - log_likelihood_current + 
                         log_prior_proposed - log_prior_current)
        
        accepted = np.log(np.random.random()) < log_acceptance
        
        if self.adaptive_mcmc:
            self.recent_acceptances[param_name].append(accepted)
        
        result = proposed_val if accepted else current_val
        
        self.timing_stats['sample_rotation'] += time.time() - t_start
        self.timing_counts['sample_rotation'] += 1
        
        return result
    
    def run_gibbs_sampler(self, verbose=True):
        """Run the optimized Gibbs sampler for real-time performance"""
        if verbose:
            print(f"\n{'='*70}")
            print(f"RUNNING REAL-TIME OPTIMIZED GIBBS SAMPLER")
            print(f"{'='*70}")
            print(f"  Candidate points: {len(self.source_points)} source, {len(self.target_points)} target")
            print(f"  Total samples: {self.n_samples}")
            print(f"  Burn-in: {self.burnin}")
            print(f"  Target time: < 1 second")
        
        total_start_time = time.time()
        
        # Initialize correspondences
        t_start = time.time()
        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nn.fit(self.target_points)
        _, initial_indices = nn.kneighbors(self.source_points)
        self.current_correspondences = initial_indices.flatten()
        self.timing_stats['initialization'] = time.time() - t_start
        
        # Pre-compute for faster sampling
        t_start = time.time()
        self.nn_sampler = NearestNeighbors(n_neighbors=min(20, len(self.target_points)), algorithm='kd_tree')
        self.nn_sampler.fit(self.target_points)
        self.timing_stats['nn_sampler_setup'] = time.time() - t_start
        
        accepted_rotation_samples = {'roll': 0, 'pitch': 0, 'yaw': 0}
        
        # Main Gibbs sampling loop
        sampling_start_time = time.time()
        
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
            old_roll = self.current_roll
            self.current_roll = self.sample_rotation_parameter(
                'roll', self.current_tx, self.current_ty, self.current_tz,
                self.current_roll, self.current_pitch, self.current_yaw,
                self.current_correspondences
            )
            if self.current_roll != old_roll:
                accepted_rotation_samples['roll'] += 1
            
            old_pitch = self.current_pitch
            self.current_pitch = self.sample_rotation_parameter(
                'pitch', self.current_tx, self.current_ty, self.current_tz,
                self.current_roll, self.current_pitch, self.current_yaw,
                self.current_correspondences
            )
            if self.current_pitch != old_pitch:
                accepted_rotation_samples['pitch'] += 1
            
            old_yaw = self.current_yaw
            self.current_yaw = self.sample_rotation_parameter(
                'yaw', self.current_tx, self.current_ty, self.current_tz,
                self.current_roll, self.current_pitch, self.current_yaw,
                self.current_correspondences
            )
            if self.current_yaw != old_yaw:
                accepted_rotation_samples['yaw'] += 1
            
            # Adapt proposal distributions
            self.adapt_proposal_stds(iteration)
            
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
        
        sampling_time = time.time() - sampling_start_time
        total_execution_time = time.time() - total_start_time
        
        self.timing_stats['total_sampling'] = sampling_time
        self.timing_stats['total_execution'] = total_execution_time
        
        # Print timing statistics
        if verbose:
            self.print_timing_statistics(accepted_rotation_samples)
            self.print_posterior_summary()
        
        return True
    
    def print_timing_statistics(self, accepted_rotation_samples):
        """Print comprehensive timing statistics"""
        print(f"\n{'='*70}")
        print(f"TIMING STATISTICS (Real-Time Performance)")
        print(f"{'='*70}")
        
        total_time = self.timing_stats['total_execution']
        total_iterations = self.n_samples + self.burnin
        
        print(f"\nOverall Performance:")
        print(f"  Total execution time: {total_time:.4f}s")
        print(f"  Total sampling time: {self.timing_stats['total_sampling']:.4f}s")
        print(f"  Time per iteration: {self.timing_stats['total_sampling']/total_iterations:.4f}s")
        print(f"  Iterations per second: {total_iterations/self.timing_stats['total_sampling']:.1f}")
        
        print(f"\nInitialization:")
        print(f"  Data loading: {self.timing_stats['data_loading']:.4f}s")
        print(f"  Candidate selection: {self.timing_stats['candidate_selection']:.4f}s")
        print(f"  NN sampler setup: {self.timing_stats['nn_sampler_setup']:.4f}s")
        print(f"  Initial correspondences: {self.timing_stats['initialization']:.4f}s")
        
        print(f"\nPer-Operation Timings (average):")
        operations = [
            ('sample_correspondences', 'Correspondence sampling'),
            ('sample_translation', 'Translation sampling'),
            ('sample_rotation', 'Rotation sampling'),
            ('log_likelihood', 'Log likelihood computation')
        ]
        
        for key, label in operations:
            if self.timing_counts[key] > 0:
                avg_time = self.timing_stats[key] / self.timing_counts[key]
                total_pct = (self.timing_stats[key] / self.timing_stats['total_sampling']) * 100
                print(f"  {label:30s}: {avg_time*1000:.3f}ms (total: {total_pct:.1f}%)")
        
        print(f"\nAcceptance Rates:")
        for param in ['roll', 'pitch', 'yaw']:
            rate = accepted_rotation_samples[param] / total_iterations
            print(f"  {param:10s}: {rate:.3f}")
        
        # Real-time performance assessment
        print(f"\n{'='*70}")
        if total_time < 1.0:
            print(f"REAL-TIME CAPABLE: {total_time:.3f}s < 1.0s target")
            fps = 1.0 / total_time
            print(f"  Theoretical max FPS: {fps:.1f} Hz")
        elif total_time < 2.0:
            print(f"NEAR REAL-TIME: {total_time:.3f}s (acceptable for some applications)")
        else:
            print(f"NOT REAL-TIME: {total_time:.3f}s > 1.0s target")
        print(f"{'='*70}")
        
        # Speedup calculation
        full_cloud_estimate = total_time * (len(self.full_source_points) / len(self.source_points))
        speedup = full_cloud_estimate / total_time
        print(f"\nEstimated Speedup vs Full Point Cloud:")
        print(f"  Current time: {total_time:.3f}s")
        print(f"  Estimated full cloud time: {full_cloud_estimate:.3f}s")
        print(f"  Speedup factor: {speedup:.1f}x")
    
    def print_posterior_summary(self):
        """Print summary statistics of posterior samples and compare with ground truth"""
        print(f"\n{'='*70}")
        print(f"POSTERIOR SUMMARY & GROUND TRUTH COMPARISON")
        print(f"{'='*70}")
        
        # Get estimates
        tx_est, ty_est, tz_est, roll_est, pitch_est, yaw_est = self.get_posterior_estimates()
        
        # Translation parameters
        print(f"\nTranslation Parameters:")
        print(f"{'Param':<8} {'True':<12} {'Estimated':<12} {'Error':<12} {'Std Dev':<12}")
        print(f"{'-'*60}")
        
        for param, true_val in [('tx', self.true_tx), ('ty', self.true_ty), ('tz', self.true_tz)]:
            samples = np.array(self.samples[param])
            est_val = np.mean(samples)
            error = est_val - true_val
            std_val = np.std(samples)
            print(f"{param.upper():<8} {true_val:<12.4f} {est_val:<12.4f} {error:<12.4f} {std_val:<12.4f}")
            print(f"         95% CI: [{np.percentile(samples, 2.5):.4f}, {np.percentile(samples, 97.5):.4f}]")
        
        # Rotation parameters
        print(f"\nRotation Parameters (degrees):")
        print(f"{'Param':<8} {'True':<12} {'Estimated':<12} {'Error':<12} {'Std Dev':<12}")
        print(f"{'-'*60}")
        
        for param, true_val in [('roll', self.true_roll), ('pitch', self.true_pitch), ('yaw', self.true_yaw)]:
            samples_rad = np.array(self.samples[param])
            samples_deg = np.degrees(samples_rad)
            true_deg = np.degrees(true_val)
            est_deg = np.mean(samples_deg)
            error_deg = est_deg - true_deg
            std_deg = np.std(samples_deg)
            print(f"{param.upper():<8} {true_deg:<12.2f} {est_deg:<12.2f} {error_deg:<12.2f} {std_deg:<12.2f}")
            print(f"         95% CI: [{np.percentile(samples_deg, 2.5):.2f}°, {np.percentile(samples_deg, 97.5):.2f}°]")
        
        # Overall error metrics
        trans_error = np.sqrt((tx_est - self.true_tx)**2 + (ty_est - self.true_ty)**2 + (tz_est - self.true_tz)**2)
        rot_error = np.sqrt((roll_est - self.true_roll)**2 + (pitch_est - self.true_pitch)**2 + (yaw_est - self.true_yaw)**2)
        
        print(f"\n{'='*70}")
        print(f"OVERALL RECOVERY QUALITY:")
        print(f"  Translation error (L2): {trans_error:.6f}")
        print(f"  Rotation error (L2): {rot_error:.6f} rad ({np.degrees(rot_error):.4f}°)")
        print(f"{'='*70}")
    
    def get_posterior_estimates(self):
        """Get point estimates from posterior"""
        tx_mean = np.mean(self.samples['tx'])
        ty_mean = np.mean(self.samples['ty'])
        tz_mean = np.mean(self.samples['tz'])
        roll_mean = np.mean(self.samples['roll'])
        pitch_mean = np.mean(self.samples['pitch'])
        yaw_mean = np.mean(self.samples['yaw'])
        
        return tx_mean, ty_mean, tz_mean, roll_mean, pitch_mean, yaw_mean
    
    def evaluate_on_full_cloud(self, verbose=True):
        """
        Evaluate the registration on the full point clouds
        (not just candidate points)
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"EVALUATING ON FULL POINT CLOUDS")
            print(f"{'='*70}")
        
        t_start = time.time()
        
        # Get transformation estimate
        tx, ty, tz, roll, pitch, yaw = self.get_posterior_estimates()
        
        # Apply to full source cloud
        transformed_full_source = self.apply_transformation(
            self.full_source_points, tx, ty, tz, roll, pitch, yaw
        )
        
        # Compute distances to target (use sampling for large clouds)
        n_eval = min(5000, len(self.full_source_points))
        eval_indices = np.random.choice(len(self.full_source_points), n_eval, replace=False)
        
        eval_source = transformed_full_source[eval_indices]
        
        # Find nearest neighbors in target
        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nn.fit(self.full_target_points)
        distances, _ = nn.kneighbors(eval_source)
        distances = distances.flatten()
        
        eval_time = time.time() - t_start
        
        # Compute metrics
        mean_error = np.mean(distances)
        median_error = np.median(distances)
        std_error = np.std(distances)
        
        if verbose:
            print(f"\nFull Cloud Registration Quality:")
            print(f"  Evaluation points: {n_eval} / {len(self.full_source_points)}")
            print(f"  Mean error: {mean_error:.6f}")
            print(f"  Median error: {median_error:.6f}")
            print(f"  Std error: {std_error:.6f}")
            print(f"  95th percentile error: {np.percentile(distances, 95):.6f}")
            print(f"  Points with error < 0.01: {np.sum(distances < 0.01)/len(distances)*100:.1f}%")
            print(f"  Points with error < 0.005: {np.sum(distances < 0.005)/len(distances)*100:.1f}%")
            print(f"  Evaluation time: {eval_time:.3f}s")
        
        return {
            'mean_error': mean_error,
            'median_error': median_error,
            'std_error': std_error,
            'percentile_95': np.percentile(distances, 95),
            'eval_time': eval_time
        }
    
    def visualize_candidate_selection(self):
        """Visualize the selected candidate points"""
        fig = plt.figure(figsize=(20, 10))
        
        # Source cloud
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot all source points in light color
        ax1.scatter(self.full_source_points[:, 0], 
                   self.full_source_points[:, 1],
                   self.full_source_points[:, 2],
                   c='lightgray', alpha=0.2, s=5, label='All points')
        
        # Plot selected candidates in bright color
        ax1.scatter(self.source_points[:, 0],
                   self.source_points[:, 1],
                   self.source_points[:, 2],
                   c='red', alpha=0.8, s=30, label='Candidates', edgecolors='darkred')
        
        # Plot centroid
        source_centroid = np.mean(self.full_source_points, axis=0)
        ax1.scatter([source_centroid[0]], [source_centroid[1]], [source_centroid[2]],
                   c='blue', s=200, marker='*', label='Centroid', edgecolors='darkblue')
        
        ax1.set_title(f'Source Point Candidates\n{len(self.source_points)} / {len(self.full_source_points)} points ({self.candidate_ratio*100:.1f}%)',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        ax1.view_init(elev=20, azim=45)
        
        # Target cloud
        ax2 = fig.add_subplot(122, projection='3d')
        
        ax2.scatter(self.full_target_points[:, 0],
                   self.full_target_points[:, 1],
                   self.full_target_points[:, 2],
                   c='lightgray', alpha=0.2, s=5, label='All points')
        
        ax2.scatter(self.target_points[:, 0],
                   self.target_points[:, 1],
                   self.target_points[:, 2],
                   c='green', alpha=0.8, s=30, label='Candidates', edgecolors='darkgreen')
        
        target_centroid = np.mean(self.full_target_points, axis=0)
        ax2.scatter([target_centroid[0]], [target_centroid[1]], [target_centroid[2]],
                   c='blue', s=200, marker='*', label='Centroid', edgecolors='darkblue')
        
        ax2.set_title(f'Target Point Candidates\n{len(self.target_points)} / {len(self.full_target_points)} points ({self.candidate_ratio*100:.1f}%)',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        ax2.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig('/Users/waelbenamara/Desktop/Research/RandomWalk/data/candidate_selection_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nCandidate selection visualization displayed and saved!")
    
    def visualize_registration(self):
        """Visualize registration result on full point clouds"""
        fig = plt.figure(figsize=(24, 12))
        
        # Before registration
        ax1 = fig.add_subplot(121, projection='3d')
        
        # After registration
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Get transformation
        tx, ty, tz, roll, pitch, yaw = self.get_posterior_estimates()
        
        # Subsample for visualization
        n_vis = min(2000, len(self.full_source_points))
        source_indices = np.random.choice(len(self.full_source_points), n_vis, replace=False)
        target_indices = np.random.choice(len(self.full_target_points), n_vis, replace=False)
        
        source_vis = self.full_source_points[source_indices]
        target_vis = self.full_target_points[target_indices]
        
        # Apply transformation
        transformed_source = self.apply_transformation(
            source_vis, tx, ty, tz, roll, pitch, yaw
        )
        
        # BEFORE
        ax1.scatter(source_vis[:, 0], source_vis[:, 1], source_vis[:, 2],
                   c='lightblue', alpha=0.6, s=20, label='Source',
                   marker='o', edgecolors='blue', linewidth=0.2)
        
        ax1.scatter(target_vis[:, 0], target_vis[:, 1], target_vis[:, 2],
                   c='lightcoral', alpha=0.6, s=20, label='Target',
                   marker='^', edgecolors='red', linewidth=0.2)
        
        # AFTER
        ax2.scatter(target_vis[:, 0], target_vis[:, 1], target_vis[:, 2],
                   c='crimson', alpha=0.7, s=25, label='Target',
                   marker='o', edgecolors='darkred', linewidth=0.3)
        
        ax2.scatter(transformed_source[:, 0], transformed_source[:, 1], transformed_source[:, 2],
                   c='dodgerblue', alpha=0.7, s=25, label='Registered Source',
                   marker='^', edgecolors='navy', linewidth=0.3)
        
        # Add correspondence lines
        distances = cdist(transformed_source, target_vis)
        n_corr = min(50, len(transformed_source))
        corr_indices = np.random.choice(len(transformed_source), n_corr, replace=False)
        
        for i in corr_indices:
            closest_idx = np.argmin(distances[i])
            dist = distances[i, closest_idx]
            if dist < 0.015:
                ax2.plot([transformed_source[i, 0], target_vis[closest_idx, 0]],
                        [transformed_source[i, 1], target_vis[closest_idx, 1]],
                        [transformed_source[i, 2], target_vis[closest_idx, 2]],
                        'gray', alpha=0.3, linewidth=0.8)
        
        # Compute errors
        all_distances = cdist(transformed_source, target_vis)
        min_distances = np.min(all_distances, axis=1)
        mean_error = np.mean(min_distances)
        
        initial_distances = cdist(source_vis, target_vis)
        initial_min_distances = np.min(initial_distances, axis=1)
        initial_mean_error = np.mean(initial_min_distances)
        
        improvement_factor = initial_mean_error / mean_error if mean_error > 0 else float('inf')
        
        # Compute transformation errors
        trans_error = np.sqrt((tx - self.true_tx)**2 + (ty - self.true_ty)**2 + (tz - self.true_tz)**2)
        rot_error = np.sqrt((roll - self.true_roll)**2 + (pitch - self.true_pitch)**2 + (yaw - self.true_yaw)**2)
        
        # Titles
        ax1.set_title(f'BEFORE Registration\n'
                     f'Mean Distance Error: {initial_mean_error:.4f}',
                     fontsize=14, fontweight='bold', color='darkred')
        
        ax2.set_title(f'AFTER Registration (Transformation Recovery)\n'
                     f'Estimated: T=({tx:.3f}, {ty:.3f}, {tz:.3f}) '
                     f'R=({np.degrees(roll):.1f}°, {np.degrees(pitch):.1f}°, {np.degrees(yaw):.1f}°)\n'
                     f'True: T=({self.true_tx:.3f}, {self.true_ty:.3f}, {self.true_tz:.3f}) '
                     f'R=({np.degrees(self.true_roll):.1f}°, {np.degrees(self.true_pitch):.1f}°, {np.degrees(self.true_yaw):.1f}°)\n'
                     f'Trans Error: {trans_error:.5f}, Rot Error: {np.degrees(rot_error):.3f}° | '
                     f'Mean Reg Error: {mean_error:.4f}',
                     fontsize=12, fontweight='bold', color='darkgreen')
        
        # Styling
        for ax in [ax1, ax2]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.grid(True, alpha=0.3)
            ax.view_init(elev=20, azim=45)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('/Users/waelbenamara/Desktop/Research/RandomWalk/data/realtime_registration_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nRegistration visualization displayed and saved!")
    
    def save_results(self, filename_prefix="realtime_gibbs_3d"):
        """Save results"""
        # Save posterior samples
        samples_df = pd.DataFrame({
            'tx': self.samples['tx'],
            'ty': self.samples['ty'],
            'tz': self.samples['tz'],
            'roll': self.samples['roll'],
            'pitch': self.samples['pitch'],
            'yaw': self.samples['yaw'],
            'roll_degrees': np.degrees(self.samples['roll']),
            'pitch_degrees': np.degrees(self.samples['pitch']),
            'yaw_degrees': np.degrees(self.samples['yaw']),
            'log_likelihood': self.samples['log_likelihood']
        })
        samples_df.to_csv(f"/Users/waelbenamara/Desktop/Research/RandomWalk/data/{filename_prefix}_samples.csv", index=False)
        
        # Save timing statistics
        timing_df = pd.DataFrame([
            {'operation': key, 'total_time': value, 'count': self.timing_counts.get(key, 1)}
            for key, value in self.timing_stats.items()
        ])
        timing_df['avg_time'] = timing_df['total_time'] / timing_df['count']
        timing_df.to_csv(f"/Users/waelbenamara/Desktop/Research/RandomWalk/data/{filename_prefix}_timing.csv", index=False)
        
        print(f"\nResults saved as {filename_prefix}_samples.csv and {filename_prefix}_timing.csv")


def compare_with_original():
    """Compare real-time optimized version with different candidate ratios"""
    print("\n" + "="*80)
    print("COMPARISON: Different Candidate Ratios")
    print("="*80)
    
    # Check if data exists
    data_dir = "/Users/waelbenamara/Desktop/Research/RandomWalk/data"
    point_cloud_file = f"{data_dir}/bunny_source.csv"
    
    if not os.path.exists(point_cloud_file):
        print(f"ERROR: Point cloud not found at {point_cloud_file}")
        return None
    
    # Test different candidate ratios
    ratios = [0.02, 0.05, 0.10, 0.20]
    
    # Define a test transformation
    test_tx, test_ty, test_tz = 0.1, 0.05, -0.08
    test_roll, test_pitch, test_yaw = 0.15, -0.1, 0.2
    
    results = []
    
    for ratio in ratios:
        print(f"\n{'='*80}")
        print(f"Testing with {ratio*100:.1f}% candidate points")
        print(f"{'='*80}")
        
        sampler = RealTimeGibbsSampler3D(
            point_cloud_file,
            true_tx=test_tx, true_ty=test_ty, true_tz=test_tz,
            true_roll=test_roll, true_pitch=test_pitch, true_yaw=test_yaw,
            candidate_ratio=ratio,
            extremity_ratio=0.9,
            add_noise=True,
            noise_std=0.001
        )
        
        sampler.run_gibbs_sampler(verbose=False)
        
        # Evaluate on full cloud
        eval_results = sampler.evaluate_on_full_cloud(verbose=False)
        
        # Get transformation errors
        tx_est, ty_est, tz_est, roll_est, pitch_est, yaw_est = sampler.get_posterior_estimates()
        trans_error = np.sqrt((tx_est - test_tx)**2 + (ty_est - test_ty)**2 + (tz_est - test_tz)**2)
        rot_error = np.degrees(np.sqrt((roll_est - test_roll)**2 + (pitch_est - test_pitch)**2 + (yaw_est - test_yaw)**2))
        
        results.append({
            'ratio': ratio,
            'n_candidates': len(sampler.source_points),
            'total_time': sampler.timing_stats['total_execution'],
            'sampling_time': sampler.timing_stats['total_sampling'],
            'mean_error': eval_results['mean_error'],
            'trans_error': trans_error,
            'rot_error': rot_error
        })
        
        print(f"  Time: {sampler.timing_stats['total_execution']:.3f}s")
        print(f"  Mean registration error: {eval_results['mean_error']:.6f}")
        print(f"  Translation recovery error: {trans_error:.6f}")
        print(f"  Rotation recovery error: {rot_error:.4f}°")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Ratio':>8} | {'Candidates':>10} | {'Time (s)':>10} | {'Trans Err':>11} | {'Rot Err':>10} | {'Speedup':>8}")
    print(f"{'-'*80}")
    
    base_time = results[0]['total_time'] * (100 / results[0]['ratio'])
    
    for r in results:
        speedup = base_time / r['total_time']
        print(f"{r['ratio']*100:>7.1f}% | {r['n_candidates']:>10d} | {r['total_time']:>10.3f} | {r['trans_error']:>11.6f} | {r['rot_error']:>9.4f}° | {speedup:>7.1f}x")
    
    return results


def test_transformation_recovery(point_cloud_file, 
                                  true_tx=0.1, true_ty=0.05, true_tz=-0.08,
                                  true_roll=0.15, true_pitch=-0.1, true_yaw=0.2,
                                  candidate_ratio=0.1, extremity_ratio=0.7,
                                  add_noise=True, noise_std=0.001,
                                  visualize=True):
    """
    Test transformation recovery with specified parameters
    
    Args:
        point_cloud_file: Path to point cloud CSV file
        true_tx, true_ty, true_tz: Ground truth translation
        true_roll, true_pitch, true_yaw: Ground truth rotation (radians)
        candidate_ratio: Fraction of points to use as candidates
        extremity_ratio: Fraction of candidates from extremities
        add_noise: Whether to add noise to target
        noise_std: Standard deviation of noise
        visualize: Whether to generate visualizations
    
    Returns:
        sampler object if successful, None otherwise
    """
    print(f"\n{'='*70}")
    print(f"TESTING TRANSFORMATION RECOVERY")
    print(f"{'='*70}")
    print(f"Ground Truth: T=({true_tx:.4f}, {true_ty:.4f}, {true_tz:.4f})")
    print(f"              R=({np.degrees(true_roll):.2f}°, {np.degrees(true_pitch):.2f}°, {np.degrees(true_yaw):.2f}°)")
    print(f"Candidates: {candidate_ratio*100:.1f}% ({extremity_ratio*100:.0f}% from extremities)")
    print(f"Noise: {'Yes' if add_noise else 'No'} (σ={noise_std if add_noise else 0})")
    
    # Create sampler
    sampler = RealTimeGibbsSampler3D(
        point_cloud_file,
        true_tx=true_tx, true_ty=true_ty, true_tz=true_tz,
        true_roll=true_roll, true_pitch=true_pitch, true_yaw=true_yaw,
        candidate_ratio=candidate_ratio,
        extremity_ratio=extremity_ratio,
        add_noise=add_noise,
        noise_std=noise_std
    )
    
    # Run sampler
    success = sampler.run_gibbs_sampler(verbose=False)
    
    if success:
        # Get estimates
        tx_est, ty_est, tz_est, roll_est, pitch_est, yaw_est = sampler.get_posterior_estimates()
        
        # Compute errors
        trans_error = np.sqrt((tx_est - true_tx)**2 + (ty_est - true_ty)**2 + (tz_est - true_tz)**2)
        rot_error = np.sqrt((roll_est - true_roll)**2 + (pitch_est - true_pitch)**2 + (yaw_est - true_yaw)**2)
        
        print(f"\n{'='*70}")
        print(f"RECOVERY RESULTS:")
        print(f"  Estimated: T=({tx_est:.4f}, {ty_est:.4f}, {tz_est:.4f})")
        print(f"             R=({np.degrees(roll_est):.2f}°, {np.degrees(pitch_est):.2f}°, {np.degrees(yaw_est):.2f}°)")
        print(f"  Translation error: {trans_error:.6f}")
        print(f"  Rotation error: {np.degrees(rot_error):.4f}°")
        print(f"  Time: {sampler.timing_stats['total_execution']:.4f}s")
        print(f"{'='*70}")
        
        if visualize:
            sampler.visualize_registration()
        
        return sampler
    else:
        print("ERROR: Transformation recovery failed")
        return None


def main():
    """Simple main function - test transformation recovery"""
    print("="*70)
    print("REAL-TIME 3D POINT CLOUD REGISTRATION")
    print("Testing Transformation Recovery")
    print("="*70)
    
    # Check if data exists
    data_dir = "/Users/waelbenamara/Desktop/Research/RandomWalk/data"
    point_cloud_file = f"{data_dir}/bunny_source.csv"
    
    if not os.path.exists(point_cloud_file):
        print("ERROR: Point cloud not found!")
        print(f"Please ensure bunny_source.csv exists in {data_dir}/ directory")
        return None
    
    print("\n1. INITIALIZING REAL-TIME SAMPLER")
    print("   Using default parameters: 10% candidate points, 70% from extremities")
    
    # Define ground truth transformation
    true_tx, true_ty, true_tz = 0.9, 0.05, -0.08
    true_roll, true_pitch, true_yaw = 1.0, -0.1, 0.2
    
    print(f"\n   Ground Truth Transformation:")
    print(f"      Translation: ({true_tx:.4f}, {true_ty:.4f}, {true_tz:.4f})")
    print(f"      Rotation: ({np.degrees(true_roll):.2f}°, {np.degrees(true_pitch):.2f}°, {np.degrees(true_yaw):.2f}°)")
    
    # Create sampler
    sampler = RealTimeGibbsSampler3D(
        point_cloud_file,
        true_tx=true_tx, true_ty=true_ty, true_tz=true_tz,
        true_roll=true_roll, true_pitch=true_pitch, true_yaw=true_yaw,
        candidate_ratio=0.1,   # 10% of points
        extremity_ratio=0.7,   # 70% from extremities
        add_noise=True,
        noise_std=0.001
    )
    
    print("\n2. VISUALIZING CANDIDATE SELECTION")
    sampler.visualize_candidate_selection()
    
    print("\n3. RUNNING TRANSFORMATION RECOVERY")
    print("   Running Gibbs sampler to recover transformation...")
    
    # Measure the transformation recovery time
    frame_start_time = time.time()
    
    # Run the registration
    success = sampler.run_gibbs_sampler(verbose=True)
    
    frame_end_time = time.time()
    transformation_time = frame_end_time - frame_start_time
    
    if success:
        print("\n4. TRANSFORMATION RECOVERY RESULTS")
        print("="*70)
        print(f"TRANSFORMATION RECOVERED SUCCESSFULLY")
        print(f"RECOVERY TIME: {transformation_time:.4f} seconds")
        print(f"RECOVERY FPS: {1.0/transformation_time:.2f} FPS")
        
        if transformation_time < 1.0:
            print(f"REAL-TIME CAPABLE: < 1.0s per transformation")
        else:
            print(f"NOT REAL-TIME: > 1.0s per transformation")
        
        # Get transformation results and compare with ground truth
        tx_est, ty_est, tz_est, roll_est, pitch_est, yaw_est = sampler.get_posterior_estimates()
        
        trans_error = np.sqrt((tx_est - true_tx)**2 + (ty_est - true_ty)**2 + (tz_est - true_tz)**2)
        rot_error = np.sqrt((roll_est - true_roll)**2 + (pitch_est - true_pitch)**2 + (yaw_est - true_yaw)**2)
        
        print(f"\nGROUND TRUTH vs ESTIMATED:")
        print(f"   Translation: True=({true_tx:.4f}, {true_ty:.4f}, {true_tz:.4f})")
        print(f"                Est=({tx_est:.4f}, {ty_est:.4f}, {tz_est:.4f})")
        print(f"                Error={trans_error:.6f}")
        print(f"   Rotation: True=({np.degrees(true_roll):.2f}°, {np.degrees(true_pitch):.2f}°, {np.degrees(true_yaw):.2f}°)")
        print(f"             Est=({np.degrees(roll_est):.2f}°, {np.degrees(pitch_est):.2f}°, {np.degrees(yaw_est):.2f}°)")
        print(f"             Error={np.degrees(rot_error):.4f}°")
        
        # Evaluate on full cloud
        print(f"\n5. EVALUATING ON FULL POINT CLOUD")
        eval_results = sampler.evaluate_on_full_cloud(verbose=False)
        print(f"   Mean registration error: {eval_results['mean_error']:.6f}")
        print(f"   Evaluation time: {eval_results['eval_time']:.4f}s")
        
        # Generate visualization
        print(f"\n6. GENERATING REGISTRATION VISUALIZATION")
        sampler.visualize_registration()
        
        print(f"\n" + "="*70)
        print(f"TRANSFORMATION RECOVERY COMPLETE!")
        print(f"Processing time: {transformation_time:.4f}s ({1.0/transformation_time:.1f} FPS)")
        print(f"Translation error: {trans_error:.6f}")
        print(f"Rotation error: {np.degrees(rot_error):.4f}°")
        print(f"="*70)
        
        return sampler
    else:
        print("Transformation recovery failed")
        return None


if __name__ == "__main__":
    print("Real-Time 3D Point Cloud Registration")
    print("Transformation Recovery Test\n")
    
    sampler = main()
    
    if sampler:
        print("\nSUCCESS: Transformation recovery completed!")
    else:
        print("\nFAILED: Could not recover transformation")