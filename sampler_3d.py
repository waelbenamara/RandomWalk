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

class GibbsSampler3DRegistration:
    """
    Gibbs Sampler for Bayesian 3D Point Cloud Registration
    
    Bayesian model:
    - Priors: tx, ty, tz ~ N(0, σ_t²), roll, pitch, yaw ~ N(0, σ_rot²)
    - Likelihood: transformed_points ~ N(target_correspondences, σ_noise²)
    - Correspondence variables z_i indicating which target point corresponds to source point i
    """
    
    def __init__(self, source_file, target_file, use_multires=True):
        self.source_points = pd.read_csv(source_file).values
        self.target_points = pd.read_csv(target_file).values
        self.use_multires = use_multires
        
        # Ensure 3D points
        if self.source_points.shape[1] != 3 or self.target_points.shape[1] != 3:
            raise ValueError("Input point clouds must be 3D (have 3 columns: x, y, z)")
        
        print(f"Loaded {len(self.source_points)} source and {len(self.target_points)} target 3D points")
        
        # Model hyperparameters
        self.setup_priors()
        
        # MCMC parameters (optimized for speed)
        self.n_samples = 2000  # Number of samples to keep
        self.burnin = 200     # Number of burn-in iterations
        self.thin = 1         # Thinning factor
        
        # Current state - 3D transformation parameters
        self.current_tx = 0.0
        self.current_ty = 0.0
        self.current_tz = 0.0
        self.current_roll = 0.0   # Rotation around x-axis
        self.current_pitch = 0.0  # Rotation around y-axis
        self.current_yaw = 0.0    # Rotation around z-axis
        self.current_correspondences = np.zeros(len(self.source_points), dtype=int)
        
        # Storage for samples
        self.samples = {
            'tx': [], 'ty': [], 'tz': [],
            'roll': [], 'pitch': [], 'yaw': [],
            'log_likelihood': [],
            'correspondences': [],
            'coverage_scores': []
        }
        
        # Coverage reinforcement parameters
        self.use_coverage_reinforcement = True
        self.coverage_weight = 0.1  # Weight for coverage term in likelihood
        
        # Adaptive MCMC parameters
        self.adaptive_mcmc = True
        self.adaptation_interval = 50  # Adapt every N iterations
        self.target_acceptance_rate = 0.44  # Optimal for multivariate sampling
        self.adaptation_factor = 1.01  # How aggressively to adapt
        
        # Proposal standard deviations (will be adapted)
        self.proposal_stds = {
            'roll': 0.05, 'pitch': 0.05, 'yaw': 0.1
        }
        
        # Track acceptance rates for adaptation
        self.recent_acceptances = {
            'roll': [], 'pitch': [], 'yaw': []
        }
        
        # Multi-resolution setup
        if self.use_multires:
            self.setup_multiresolution()
        
    def setup_priors(self):
        """Setup adaptive prior distributions and hyperparameters for 3D"""
        # Analyze data to set reasonable priors
        source_center = np.mean(self.source_points, axis=0)
        target_center = np.mean(self.target_points, axis=0)
        center_diff = target_center - source_center
        
        # Compute data scale for adaptive priors
        source_scale = np.std(self.source_points, axis=0)
        target_scale = np.std(self.target_points, axis=0)
        data_scale = np.mean([source_scale, target_scale], axis=0)
        overall_scale = np.mean(data_scale)
        
        # Translation prior parameters (adaptive to data scale)
        self.prior_tx_mean = center_diff[0]
        self.prior_tx_var = (3 * overall_scale)**2  # 3 standard deviations of data
        
        self.prior_ty_mean = center_diff[1] 
        self.prior_ty_var = (3 * overall_scale)**2
        
        self.prior_tz_mean = center_diff[2]
        self.prior_tz_var = (3 * overall_scale)**2
        
        # Rotation prior parameters (more informative for stability)
        self.prior_roll_mean = 0.0
        self.prior_roll_var = (np.pi/3)**2  # ±60 degrees
        
        self.prior_pitch_mean = 0.0
        self.prior_pitch_var = (np.pi/3)**2
        
        self.prior_yaw_mean = 0.0
        self.prior_yaw_var = (np.pi/3)**2
        
        # Adaptive noise parameter based on data density
        nn_distances = self.compute_nearest_neighbor_distances()
        median_nn_distance = np.median(nn_distances)
        self.noise_precision = 1.0 / (median_nn_distance**2)  # Adaptive to point density
        
        print(f"3D Prior setup:")
        print(f"  TX prior: N({self.prior_tx_mean:.2f}, {np.sqrt(self.prior_tx_var):.2f})")
        print(f"  TY prior: N({self.prior_ty_mean:.2f}, {np.sqrt(self.prior_ty_var):.2f})")
        print(f"  TZ prior: N({self.prior_tz_mean:.2f}, {np.sqrt(self.prior_tz_var):.2f})")
        print(f"  Roll prior: N({self.prior_roll_mean:.2f}, {np.sqrt(self.prior_roll_var):.2f})")
        print(f"  Pitch prior: N({self.prior_pitch_mean:.2f}, {np.sqrt(self.prior_pitch_var):.2f})")
        print(f"  Yaw prior: N({self.prior_yaw_mean:.2f}, {np.sqrt(self.prior_yaw_var):.2f})")
        print(f"  Noise precision: {self.noise_precision:.2f} (σ = {1/np.sqrt(self.noise_precision):.3f})")
        print(f"  Data scale: {overall_scale:.4f}, NN distance: {median_nn_distance:.4f}")
        
    def compute_nearest_neighbor_distances(self):
        """Compute nearest neighbor distances for adaptive noise estimation"""
        # Use a subset for speed
        n_sample = min(1000, len(self.source_points))
        indices = np.random.choice(len(self.source_points), n_sample, replace=False)
        sample_points = self.source_points[indices]
        
        nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')  # k=2 to get 1st neighbor (not self)
        nn.fit(sample_points)
        distances, _ = nn.kneighbors(sample_points)
        return distances[:, 1]  # Return distances to 1st neighbor (not self)
    
    def adapt_proposal_stds(self, iteration):
        """Adapt proposal standard deviations based on acceptance rates"""
        if not self.adaptive_mcmc or iteration < self.adaptation_interval:
            return
            
        if iteration % self.adaptation_interval == 0:
            for param in ['roll', 'pitch', 'yaw']:
                if len(self.recent_acceptances[param]) > 0:
                    acceptance_rate = np.mean(self.recent_acceptances[param])
                    
                    # Adapt proposal std
                    if acceptance_rate > self.target_acceptance_rate:
                        # Too many acceptances - increase step size
                        self.proposal_stds[param] *= self.adaptation_factor
                    else:
                        # Too few acceptances - decrease step size
                        self.proposal_stds[param] /= self.adaptation_factor
                    
                    # Keep within reasonable bounds
                    self.proposal_stds[param] = np.clip(self.proposal_stds[param], 0.001, 0.5)
                    
                    # Clear recent acceptances
                    self.recent_acceptances[param] = []
    
    def setup_multiresolution(self):
        """Setup multi-resolution point cloud hierarchy for coarse-to-fine registration"""
        # Create multiple resolution levels
        self.resolution_levels = []
        
        # Level 0: Coarsest (500 points)
        n_coarse = min(500, len(self.source_points) // 4)
        coarse_source_idx = np.random.choice(len(self.source_points), n_coarse, replace=False)
        coarse_target_idx = np.random.choice(len(self.target_points), n_coarse, replace=False)
        
        # Level 1: Medium (2000 points)  
        n_medium = min(2000, len(self.source_points) // 2)
        medium_source_idx = np.random.choice(len(self.source_points), n_medium, replace=False)
        medium_target_idx = np.random.choice(len(self.target_points), n_medium, replace=False)
        
        # Level 2: Fine (full resolution)
        fine_source_idx = np.arange(len(self.source_points))
        fine_target_idx = np.arange(len(self.target_points))
        
        self.resolution_levels = [
            {
                'source_idx': coarse_source_idx,
                'target_idx': coarse_target_idx,
                'n_samples': 100,  # Fewer samples for coarse level
                'burnin': 50
            },
            {
                'source_idx': medium_source_idx,
                'target_idx': medium_target_idx, 
                'n_samples': 300,  # Medium samples
                'burnin': 100
            },
            {
                'source_idx': fine_source_idx,
                'target_idx': fine_target_idx,
                'n_samples': 1000,  # Full samples
                'burnin': 200
            }
        ]
        
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
        
        # Apply rotation then translation
        rotated_points = points @ rotation_matrix.T
        transformed_points = rotated_points + np.array([tx, ty, tz])
        return transformed_points
    
    def compute_coverage_score(self, correspondences):
        """Compute coverage score - how well target points are covered"""
        # Count how many times each target point is matched
        target_counts = np.bincount(correspondences, minlength=len(self.target_points))
        
        # Coverage metrics
        covered_targets = np.sum(target_counts > 0)  # Number of covered target points
        coverage_ratio = covered_targets / len(self.target_points)  # Fraction covered
        
        # Penalty for uneven coverage (prefer uniform distribution)
        expected_count = len(self.source_points) / len(self.target_points)
        coverage_uniformity = -np.var(target_counts) / (expected_count**2 + 1e-6)
        
        return coverage_ratio, coverage_uniformity
    
    def compute_log_likelihood(self, tx, ty, tz, roll, pitch, yaw, correspondences):
        """Compute log likelihood of current state with optional coverage reinforcement"""
        transformed_source = self.apply_transformation(
            self.source_points, tx, ty, tz, roll, pitch, yaw
        )
        
        # Get corresponding target points
        target_correspondences = self.target_points[correspondences]
        
        # Compute squared distances (3D)
        squared_distances = np.sum((transformed_source - target_correspondences)**2, axis=1)
        
        # Standard log likelihood under Gaussian noise model
        log_likelihood = -0.5 * self.noise_precision * np.sum(squared_distances)
        log_likelihood -= 0.5 * len(self.source_points) * np.log(2 * np.pi / self.noise_precision)
        
        # Add coverage reinforcement term
        if self.use_coverage_reinforcement:
            coverage_ratio, coverage_uniformity = self.compute_coverage_score(correspondences)
            coverage_bonus = self.coverage_weight * (coverage_ratio + 0.1 * coverage_uniformity)
            log_likelihood += coverage_bonus
        
        return log_likelihood
    
    def sample_correspondences(self, tx, ty, tz, roll, pitch, yaw):
        """Sample correspondence variables z_i (optimized with approximate sampling)"""
        transformed_source = self.apply_transformation(
            self.source_points, tx, ty, tz, roll, pitch, yaw
        )
        
        # Use approximate sampling for speed: only consider k nearest neighbors
        k_neighbors = min(50, len(self.target_points))
        distances, neighbor_indices = self.nn_sampler.kneighbors(transformed_source)
        
        # Compute probabilities only for k nearest neighbors
        distances_squared = distances ** 2
        log_probs = -0.5 * self.noise_precision * distances_squared
        
        # Numerically stable softmax
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - max_log_probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        
        # Sample from k nearest neighbors
        cumprobs = np.cumsum(probs, axis=1)
        random_vals = np.random.random(len(transformed_source))[:, np.newaxis]
        local_indices = np.argmax(cumprobs >= random_vals, axis=1)
        
        # Map back to global indices
        new_correspondences = neighbor_indices[np.arange(len(transformed_source)), local_indices]
        
        return new_correspondences
    
    def sample_tx(self, ty, tz, roll, pitch, yaw, correspondences):
        """Sample tx given other parameters"""
        transformed_without_tx = self.apply_transformation(
            self.source_points, 0, ty, tz, roll, pitch, yaw
        )
        target_correspondences = self.target_points[correspondences]
        
        # Compute sufficient statistics
        n = len(self.source_points)
        sum_diff_x = np.sum(target_correspondences[:, 0] - transformed_without_tx[:, 0])
        
        # Posterior parameters (conjugate normal prior/likelihood)
        posterior_precision = self.noise_precision * n + 1.0 / self.prior_tx_var
        posterior_mean = (self.noise_precision * sum_diff_x + self.prior_tx_mean / self.prior_tx_var) / posterior_precision
        posterior_var = 1.0 / posterior_precision
        
        # Sample from posterior
        return np.random.normal(posterior_mean, np.sqrt(posterior_var))
    
    def sample_ty(self, tx, tz, roll, pitch, yaw, correspondences):
        """Sample ty given other parameters"""
        transformed_without_ty = self.apply_transformation(
            self.source_points, tx, 0, tz, roll, pitch, yaw
        )
        target_correspondences = self.target_points[correspondences]
        
        # Compute sufficient statistics
        n = len(self.source_points)
        sum_diff_y = np.sum(target_correspondences[:, 1] - transformed_without_ty[:, 1])
        
        # Posterior parameters
        posterior_precision = self.noise_precision * n + 1.0 / self.prior_ty_var
        posterior_mean = (self.noise_precision * sum_diff_y + self.prior_ty_mean / self.prior_ty_var) / posterior_precision
        posterior_var = 1.0 / posterior_precision
        
        # Sample from posterior
        return np.random.normal(posterior_mean, np.sqrt(posterior_var))
    
    def sample_tz(self, tx, ty, roll, pitch, yaw, correspondences):
        """Sample tz given other parameters"""
        transformed_without_tz = self.apply_transformation(
            self.source_points, tx, ty, 0, roll, pitch, yaw
        )
        target_correspondences = self.target_points[correspondences]
        
        # Compute sufficient statistics
        n = len(self.source_points)
        sum_diff_z = np.sum(target_correspondences[:, 2] - transformed_without_tz[:, 2])
        
        # Posterior parameters
        posterior_precision = self.noise_precision * n + 1.0 / self.prior_tz_var
        posterior_mean = (self.noise_precision * sum_diff_z + self.prior_tz_mean / self.prior_tz_var) / posterior_precision
        posterior_var = 1.0 / posterior_precision
        
        # Sample from posterior
        return np.random.normal(posterior_mean, np.sqrt(posterior_var))
    
    def sample_rotation_parameter(self, param_name, tx, ty, tz, roll, pitch, yaw, correspondences):
        """Sample rotation parameter using Metropolis-Hastings within Gibbs (optimized)"""
        # Get current parameter value and prior parameters
        param_map = {
            'roll': (roll, self.prior_roll_mean, self.prior_roll_var, 0),
            'pitch': (pitch, self.prior_pitch_mean, self.prior_pitch_var, 1), 
            'yaw': (yaw, self.prior_yaw_mean, self.prior_yaw_var, 2)
        }
        
        current_val, prior_mean, prior_var, param_idx = param_map[param_name]
        
        # Use adaptive proposal std
        proposal_std = self.proposal_stds[param_name]
        proposed_val = current_val + np.random.normal(0, proposal_std)
        
        # Create parameter arrays for efficient computation
        current_params = np.array([roll, pitch, yaw])
        proposed_params = current_params.copy()
        proposed_params[param_idx] = proposed_val
        
        # Compute log acceptance ratio (only if parameters are significantly different)
        if abs(proposed_val - current_val) < 1e-6:
            return current_val
            
        log_likelihood_current = self.compute_log_likelihood(tx, ty, tz, *current_params, correspondences)
        log_likelihood_proposed = self.compute_log_likelihood(tx, ty, tz, *proposed_params, correspondences)
        
        # Prior ratio
        log_prior_current = -0.5 * (current_val - prior_mean)**2 / prior_var
        log_prior_proposed = -0.5 * (proposed_val - prior_mean)**2 / prior_var
        
        log_acceptance = (log_likelihood_proposed - log_likelihood_current + 
                         log_prior_proposed - log_prior_current)
        
        # Accept or reject
        accepted = np.log(np.random.random()) < log_acceptance
        
        # Track acceptance for adaptation
        if self.adaptive_mcmc:
            self.recent_acceptances[param_name].append(accepted)
        
        if accepted:
            return proposed_val
        else:
            return current_val
    
    def run_gibbs_sampler(self, verbose=True):
        """Run the 3D Gibbs sampler"""
        if verbose:
            print(f"\nRunning 3D Gibbs sampler...")
            print(f"  Total samples: {self.n_samples}")
            print(f"  Burn-in: {self.burnin}")
            print(f"  Thinning: {self.thin}")
        
        start_time = time.time()
        
        # Initialize correspondences using nearest neighbors (faster initialization)
        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nn.fit(self.target_points)
        _, initial_indices = nn.kneighbors(self.source_points)
        self.current_correspondences = initial_indices.flatten()
        
        # Pre-compute for faster sampling
        self.nn_sampler = NearestNeighbors(n_neighbors=min(50, len(self.target_points)), algorithm='kd_tree')
        self.nn_sampler.fit(self.target_points)
        
        accepted_rotation_samples = {'roll': 0, 'pitch': 0, 'yaw': 0}
        
        # Convergence tracking for early stopping
        recent_log_likelihoods = []
        convergence_window = 20
        convergence_threshold = 1e-3
        
        for iteration in range(self.n_samples + self.burnin):
            # Gibbs sampling steps
            
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
            
            # 3. Sample rotation parameters (Metropolis-Hastings)
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
            
            # Store samples (after burn-in and thinning)
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
                
                # Store coverage score
                if self.use_coverage_reinforcement:
                    coverage_ratio, coverage_uniformity = self.compute_coverage_score(self.current_correspondences)
                    self.samples['coverage_scores'].append(coverage_ratio)
            
            # Convergence checking and progress reporting
            if iteration >= self.burnin:
                current_log_likelihood = self.compute_log_likelihood(
                    self.current_tx, self.current_ty, self.current_tz,
                    self.current_roll, self.current_pitch, self.current_yaw,
                    self.current_correspondences
                )
                recent_log_likelihoods.append(current_log_likelihood)
                
                # Check for convergence (early stopping)
                if len(recent_log_likelihoods) >= convergence_window:
                    recent_log_likelihoods = recent_log_likelihoods[-convergence_window:]
                    if len(recent_log_likelihoods) == convergence_window:
                        log_lik_std = np.std(recent_log_likelihoods)
                        if log_lik_std < convergence_threshold:
                            if verbose:
                                print(f"  Convergence detected at iteration {iteration} (log-likelihood std: {log_lik_std:.6f})")
                            break
            
            # Progress reporting
            if verbose and iteration % 50 == 0:
                if iteration < self.burnin:
                    current_log_likelihood = self.compute_log_likelihood(
                        self.current_tx, self.current_ty, self.current_tz,
                        self.current_roll, self.current_pitch, self.current_yaw,
                        self.current_correspondences
                    )
                print(f"  Iteration {iteration}: TX={self.current_tx:.3f}, TY={self.current_ty:.3f}, TZ={self.current_tz:.3f}")
                print(f"    Roll={np.degrees(self.current_roll):.1f}°, Pitch={np.degrees(self.current_pitch):.1f}°, Yaw={np.degrees(self.current_yaw):.1f}°")
                print(f"    LogLik={current_log_likelihood:.1f}")
        
        execution_time = time.time() - start_time
        total_iterations = self.n_samples + self.burnin
        
        if verbose:
            print(f"\n3D Gibbs sampling completed:")
            print(f"  Total time: {execution_time:.2f}s")
            print(f"  Effective samples: {len(self.samples['tx'])}")
            print(f"  Rotation acceptance rates:")
            for param in ['roll', 'pitch', 'yaw']:
                rate = accepted_rotation_samples[param] / total_iterations
                print(f"    {param}: {rate:.3f}")
            
            # Posterior summaries
            self.print_posterior_summary()
        
        return True
    
    def print_posterior_summary(self):
        """Print summary statistics of posterior samples"""
        print(f"\n3D Posterior Summary:")
        
        # Translation parameters
        for param in ['tx', 'ty', 'tz']:
            samples = np.array(self.samples[param])
            print(f"  {param.upper()}: mean={np.mean(samples):.4f}, std={np.std(samples):.4f}")
            print(f"      95% CI: [{np.percentile(samples, 2.5):.4f}, {np.percentile(samples, 97.5):.4f}]")
        
        # Rotation parameters
        for param in ['roll', 'pitch', 'yaw']:
            samples_rad = np.array(self.samples[param])
            samples_deg = np.degrees(samples_rad)
            print(f"  {param.upper()}: mean={np.mean(samples_deg):.2f}°, std={np.std(samples_deg):.2f}°")
            print(f"      95% CI: [{np.percentile(samples_deg, 2.5):.2f}°, {np.percentile(samples_deg, 97.5):.2f}°]")
    
    def get_posterior_estimates(self):
        """Get point estimates from posterior"""
        tx_mean = np.mean(self.samples['tx'])
        ty_mean = np.mean(self.samples['ty'])
        tz_mean = np.mean(self.samples['tz'])
        roll_mean = np.mean(self.samples['roll'])
        pitch_mean = np.mean(self.samples['pitch'])
        yaw_mean = np.mean(self.samples['yaw'])
        
        return tx_mean, ty_mean, tz_mean, roll_mean, pitch_mean, yaw_mean
    
    def plot_trace_plots(self):
        """Plot MCMC trace plots for 3D parameters"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Translation traces
        for i, param in enumerate(['tx', 'ty', 'tz']):
            axes[0, i].plot(self.samples[param])
            axes[0, i].set_title(f'{param.upper()} Trace Plot')
            axes[0, i].set_xlabel('Iteration')
            axes[0, i].set_ylabel(param.upper())
            axes[0, i].grid(True, alpha=0.3)
        
        # Rotation traces (in degrees)
        for i, param in enumerate(['roll', 'pitch', 'yaw']):
            samples_deg = np.degrees(self.samples[param])
            axes[1, i].plot(samples_deg)
            axes[1, i].set_title(f'{param.upper()} Trace Plot')
            axes[1, i].set_xlabel('Iteration')
            axes[1, i].set_ylabel(f'{param.upper()} (degrees)')
            axes[1, i].grid(True, alpha=0.3)
        
        # Log likelihood and other diagnostics
        axes[2, 0].plot(self.samples['log_likelihood'])
        axes[2, 0].set_title('Log Likelihood Trace Plot')
        axes[2, 0].set_xlabel('Iteration')
        axes[2, 0].set_ylabel('Log Likelihood')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Remove empty subplots
        axes[2, 1].remove()
        axes[2, 2].remove()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_3d_registration(self):
        """Visualize 3D registration - before and after comparison"""
        # Create figure with two subplots side by side
        fig = plt.figure(figsize=(24, 12))
        
        # Before registration (left subplot)
        ax1 = fig.add_subplot(121, projection='3d')
        
        # After registration (right subplot)  
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Get the best transformation estimate
        tx_mean, ty_mean, tz_mean, roll_mean, pitch_mean, yaw_mean = self.get_posterior_estimates()
        
        # Subsample for clearer visualization (more points for better detail)
        n_vis = min(2000, len(self.source_points))
        source_indices = np.random.choice(len(self.source_points), n_vis, replace=False)
        target_indices = np.random.choice(len(self.target_points), n_vis, replace=False)
        
        source_vis = self.source_points[source_indices]
        target_vis = self.target_points[target_indices]
        
        # Apply best transformation to source points
        transformed_source = self.apply_transformation(
            source_vis, tx_mean, ty_mean, tz_mean,
            roll_mean, pitch_mean, yaw_mean
        )
        
        # === BEFORE REGISTRATION (Left Plot) ===
        # Plot original source points (unregistered)
        ax1.scatter(source_vis[:, 0], source_vis[:, 1], source_vis[:, 2], 
                   c='lightblue', alpha=0.6, s=20, label='Original Source (bun000)', 
                   marker='o', edgecolors='blue', linewidth=0.2)
        
        # Plot target points
        ax1.scatter(target_vis[:, 0], target_vis[:, 1], target_vis[:, 2], 
                   c='lightcoral', alpha=0.6, s=20, label='Target (bun045)', 
                   marker='^', edgecolors='red', linewidth=0.2)
        
        # === AFTER REGISTRATION (Right Plot) ===
        # Plot target points (reference)
        ax2.scatter(target_vis[:, 0], target_vis[:, 1], target_vis[:, 2], 
                   c='crimson', alpha=0.7, s=25, label='Target (bun045)', 
                   marker='o', edgecolors='darkred', linewidth=0.3)
        
        # Plot transformed source points (registered)
        ax2.scatter(transformed_source[:, 0], transformed_source[:, 1], transformed_source[:, 2], 
                   c='dodgerblue', alpha=0.7, s=25, label='Registered Source (bun000)', 
                   marker='^', edgecolors='navy', linewidth=0.3)
        
        # Add correspondence lines to the AFTER plot only
        distances = cdist(transformed_source, target_vis)
        n_corr = min(50, len(transformed_source))
        corr_indices = np.random.choice(len(transformed_source), n_corr, replace=False)
        
        good_correspondences = 0
        for i in corr_indices:
            closest_idx = np.argmin(distances[i])
            dist = distances[i, closest_idx]
            if dist < 0.015:  # Only show very close correspondences
                ax2.plot([transformed_source[i, 0], target_vis[closest_idx, 0]],
                        [transformed_source[i, 1], target_vis[closest_idx, 1]], 
                        [transformed_source[i, 2], target_vis[closest_idx, 2]], 
                        'gray', alpha=0.3, linewidth=0.8)
                good_correspondences += 1
        
        # Compute registration quality metrics
        all_distances = cdist(transformed_source, target_vis)
        min_distances = np.min(all_distances, axis=1)
        mean_error = np.mean(min_distances)
        median_error = np.median(min_distances)
        
        # Compute initial misalignment for comparison
        initial_distances = cdist(source_vis, target_vis)
        initial_min_distances = np.min(initial_distances, axis=1)
        initial_mean_error = np.mean(initial_min_distances)
        
        # Set equal aspect ratio and good viewing angle for both plots
        ax1.set_box_aspect([1,1,1])
        ax2.set_box_aspect([1,1,1])
        
        # Set titles for both plots
        improvement_factor = initial_mean_error / mean_error if mean_error > 0 else float('inf')
        
        ax1.set_title(f'BEFORE Registration\n'
                     f'Stanford Bunny: Misaligned Point Clouds\n'
                     f'Mean Distance Error: {initial_mean_error:.4f}', 
                     fontsize=14, fontweight='bold', pad=20, color='darkred')
        
        ax2.set_title(f'AFTER Registration\n'
                     f'T=({tx_mean:.3f}, {ty_mean:.3f}, {tz_mean:.3f}) '
                     f'R=({np.degrees(roll_mean):.1f}°, {np.degrees(pitch_mean):.1f}°, {np.degrees(yaw_mean):.1f}°)\n'
                     f'Mean Error: {mean_error:.4f} (Improved {improvement_factor:.1f}x)', 
                     fontsize=14, fontweight='bold', pad=20, color='darkgreen')
        
        # Set axis labels for both plots
        for ax in [ax1, ax2]:
            ax.set_xlabel('X Coordinate', fontsize=12, labelpad=10)
            ax.set_ylabel('Y Coordinate', fontsize=12, labelpad=10)
            ax.set_zlabel('Z Coordinate', fontsize=12, labelpad=10)
            
            # Improve grid and background
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            # Make pane edges more subtle
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            ax.xaxis.pane.set_alpha(0.1)
            ax.yaxis.pane.set_alpha(0.1)
            ax.zaxis.pane.set_alpha(0.1)
            
            # Set optimal viewing angle for bunny
            ax.view_init(elev=20, azim=45)
        
        # Position legends
        ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10, 
                  frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10, 
                  frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed registration statistics
        print(f"\n" + "=" * 60)
        print("STANFORD BUNNY REGISTRATION RESULTS")
        print("=" * 60)
        print(f"Transformation Found:")
        print(f"  Translation: TX={tx_mean:.6f}, TY={ty_mean:.6f}, TZ={tz_mean:.6f}")
        print(f"  Rotation: Roll={np.degrees(roll_mean):.2f}°, Pitch={np.degrees(pitch_mean):.2f}°, Yaw={np.degrees(yaw_mean):.2f}°")
        
        print(f"\nRegistration Quality:")
        print(f"  Mean registration error: {mean_error:.6f}")
        print(f"  Median registration error: {median_error:.6f}")
        print(f"  95th percentile error: {np.percentile(min_distances, 95):.6f}")
        print(f"  Points with error < 0.01: {np.sum(min_distances < 0.01)/len(min_distances)*100:.1f}%")
        print(f"  Points with error < 0.005: {np.sum(min_distances < 0.005)/len(min_distances)*100:.1f}%")
        print(f"  Close correspondences shown: {good_correspondences}/{n_corr} ({good_correspondences/n_corr*100:.1f}%)")
        
        # Coverage statistics
        if self.use_coverage_reinforcement and len(self.samples['coverage_scores']) > 0:
            final_coverage = self.samples['coverage_scores'][-1]
            mean_coverage = np.mean(self.samples['coverage_scores'])
            print(f"\nTarget Coverage Statistics:")
            print(f"  Final target coverage: {final_coverage:.1%}")
            print(f"  Mean coverage during sampling: {mean_coverage:.1%}")
            print(f"  Coverage reinforcement weight: {self.coverage_weight}")
    
    def load_ground_truth(self, ground_truth_file="/Users/waelbenamara/Desktop/Research/RandomWalk/data/bunny_ground_truth.json"):
        """Load ground truth transformation if available"""
        if os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r') as f:
                return json.load(f)
        return None
    
    def evaluate_against_ground_truth(self, ground_truth):
        """Evaluate registration accuracy against ground truth"""
        if ground_truth is None:
            return None
        
        # Get posterior estimates
        tx_est, ty_est, tz_est, roll_est, pitch_est, yaw_est = self.get_posterior_estimates()
        
        # Compute errors
        translation_errors = {
            'tx': abs(tx_est - ground_truth['tx']),
            'ty': abs(ty_est - ground_truth['ty']),
            'tz': abs(tz_est - ground_truth['tz'])
        }
        
        rotation_errors = {
            'roll': abs(roll_est - ground_truth['roll']),
            'pitch': abs(pitch_est - ground_truth['pitch']),
            'yaw': abs(yaw_est - ground_truth['yaw'])
        }
        
        # Compute overall metrics
        translation_rmse = np.sqrt(np.mean([e**2 for e in translation_errors.values()]))
        rotation_rmse_rad = np.sqrt(np.mean([e**2 for e in rotation_errors.values()]))
        rotation_rmse_deg = np.degrees(rotation_rmse_rad)
        
        # Compute 95% credible interval coverage
        coverage_results = {}
        for param in ['tx', 'ty', 'tz', 'roll', 'pitch', 'yaw']:
            samples = np.array(self.samples[param])
            ci_lower = np.percentile(samples, 2.5)
            ci_upper = np.percentile(samples, 97.5)
            true_value = ground_truth[param]
            coverage_results[param] = ci_lower <= true_value <= ci_upper
        
        return {
            'translation_errors': translation_errors,
            'rotation_errors': rotation_errors,
            'translation_rmse': translation_rmse,
            'rotation_rmse_rad': rotation_rmse_rad,
            'rotation_rmse_deg': rotation_rmse_deg,
            'credible_interval_coverage': coverage_results,
            'coverage_rate': np.mean(list(coverage_results.values()))
        }
    
    def print_evaluation_results(self, ground_truth, evaluation):
        """Print detailed evaluation results"""
        if ground_truth is None or evaluation is None:
            print("\nNo ground truth available for evaluation.")
            return
        
        print(f"\n" + "=" * 70)
        print("REGISTRATION ACCURACY EVALUATION")
        print("=" * 70)
        
        # Get estimates
        tx_est, ty_est, tz_est, roll_est, pitch_est, yaw_est = self.get_posterior_estimates()
        
        print(f"\nTranslation Comparison:")
        print(f"  Parameter | Estimated    | Ground Truth | Absolute Error")
        print(f"  ----------|--------------|--------------|---------------")
        print(f"  TX        | {tx_est:11.6f} | {ground_truth['tx']:11.6f} | {evaluation['translation_errors']['tx']:.6f}")
        print(f"  TY        | {ty_est:11.6f} | {ground_truth['ty']:11.6f} | {evaluation['translation_errors']['ty']:.6f}")
        print(f"  TZ        | {tz_est:11.6f} | {ground_truth['tz']:11.6f} | {evaluation['translation_errors']['tz']:.6f}")
        
        print(f"\nRotation Comparison (degrees):")
        print(f"  Parameter | Estimated    | Ground Truth | Absolute Error")
        print(f"  ----------|--------------|--------------|---------------")
        print(f"  Roll      | {np.degrees(roll_est):11.2f}° | {np.degrees(ground_truth['roll']):11.2f}° | {np.degrees(evaluation['rotation_errors']['roll']):.2f}°")
        print(f"  Pitch     | {np.degrees(pitch_est):11.2f}° | {np.degrees(ground_truth['pitch']):11.2f}° | {np.degrees(evaluation['rotation_errors']['pitch']):.2f}°")
        print(f"  Yaw       | {np.degrees(yaw_est):11.2f}° | {np.degrees(ground_truth['yaw']):11.2f}° | {np.degrees(evaluation['rotation_errors']['yaw']):.2f}°")
        
        print(f"\nOverall Accuracy Metrics:")
        print(f"  Translation RMSE: {evaluation['translation_rmse']:.6f}")
        print(f"  Rotation RMSE: {evaluation['rotation_rmse_deg']:.2f}° ({evaluation['rotation_rmse_rad']:.6f} rad)")
        
        # Performance assessment
        print(f"\nPerformance Assessment:")
        if evaluation['translation_rmse'] < 0.01:
            trans_grade = "EXCELLENT"
        elif evaluation['translation_rmse'] < 0.05:
            trans_grade = "GOOD"
        elif evaluation['translation_rmse'] < 0.1:
            trans_grade = "FAIR"
        else:
            trans_grade = "POOR"
        
        if evaluation['rotation_rmse_deg'] < 2.0:
            rot_grade = "EXCELLENT"
        elif evaluation['rotation_rmse_deg'] < 5.0:
            rot_grade = "GOOD"
        elif evaluation['rotation_rmse_deg'] < 10.0:
            rot_grade = "FAIR"
        else:
            rot_grade = "POOR"
        
        print(f"  Translation accuracy: {trans_grade}")
        print(f"  Rotation accuracy: {rot_grade}")
        print(f"  Overall registration quality: {'EXCELLENT' if trans_grade in ['EXCELLENT', 'GOOD'] and rot_grade in ['EXCELLENT', 'GOOD'] else 'GOOD' if trans_grade != 'POOR' and rot_grade != 'POOR' else 'NEEDS IMPROVEMENT'}")
    
    def save_results(self, filename_prefix="gibbs_3d_registration", ground_truth=None, evaluation=None):
        """Save 3D Gibbs sampling results with evaluation"""
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
        
        # Save summary statistics with evaluation
        tx_mean, ty_mean, tz_mean, roll_mean, pitch_mean, yaw_mean = self.get_posterior_estimates()
        with open(f"/Users/waelbenamara/Desktop/Research/RandomWalk/data/{filename_prefix}_summary.txt", 'w') as f:
            f.write("3D Gibbs Sampler Point Cloud Registration Results\n")
            f.write("===============================================\n\n")
            f.write(f"Posterior Mean Estimates:\n")
            f.write(f"  TX: {tx_mean:.6f}\n")
            f.write(f"  TY: {ty_mean:.6f}\n")
            f.write(f"  TZ: {tz_mean:.6f}\n")
            f.write(f"  Roll: {roll_mean:.6f} radians ({np.degrees(roll_mean):.2f} degrees)\n")
            f.write(f"  Pitch: {pitch_mean:.6f} radians ({np.degrees(pitch_mean):.2f} degrees)\n")
            f.write(f"  Yaw: {yaw_mean:.6f} radians ({np.degrees(yaw_mean):.2f} degrees)\n\n")
            
            f.write(f"Posterior Standard Deviations:\n")
            for param in ['tx', 'ty', 'tz']:
                f.write(f"  {param.upper()}: {np.std(self.samples[param]):.6f}\n")
            for param in ['roll', 'pitch', 'yaw']:
                f.write(f"  {param.upper()}: {np.std(self.samples[param]):.6f} radians ({np.std(np.degrees(self.samples[param])):.2f} degrees)\n")
            
            # Add evaluation results if available
            if ground_truth is not None and evaluation is not None:
                f.write(f"\n" + "="*50 + "\n")
                f.write(f"GROUND TRUTH EVALUATION\n")
                f.write(f"="*50 + "\n\n")
                f.write(f"Translation RMSE: {evaluation['translation_rmse']:.6f}\n")
                f.write(f"Rotation RMSE: {evaluation['rotation_rmse_deg']:.2f}° ({evaluation['rotation_rmse_rad']:.6f} rad)\n")
                f.write(f"Credible Interval Coverage Rate: {evaluation['coverage_rate']:.1%}\n\n")
                
                f.write(f"Individual Parameter Errors:\n")
                for param in ['tx', 'ty', 'tz']:
                    f.write(f"  {param.upper()}: {evaluation['translation_errors'][param]:.6f}\n")
                for param in ['roll', 'pitch', 'yaw']:
                    f.write(f"  {param.upper()}: {np.degrees(evaluation['rotation_errors'][param]):.2f}° ({evaluation['rotation_errors'][param]:.6f} rad)\n")
        
        print(f"3D results saved as data/{filename_prefix}_samples.csv and data/{filename_prefix}_summary.txt")


def generate_3d_demo_data():
    """Generate 3D demo point clouds (spheres) for testing"""
    np.random.seed(42)
    
    # Generate points on a sphere (source)
    n_points = 100
    phi = np.random.uniform(0, 2*np.pi, n_points)  # azimuthal angle
    costheta = np.random.uniform(-1, 1, n_points)  # cos of polar angle
    theta = np.arccos(costheta)
    
    radius = 2.0
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    source_points = np.column_stack([x, y, z])
    
    # Apply known transformation to create target
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
    target_points += np.random.normal(0, 0.1, target_points.shape)  # Add noise
    
    # Save to CSV files in data directory
    pd.DataFrame(source_points, columns=['x', 'y', 'z']).to_csv('/Users/waelbenamara/Desktop/Research/RandomWalk/data/demo_3d_source.csv', index=False)
    pd.DataFrame(target_points, columns=['x', 'y', 'z']).to_csv('/Users/waelbenamara/Desktop/Research/RandomWalk/data/demo_3d_target.csv', index=False)
    
    print(f"Generated 3D demo data:")
    print(f"  Source: {len(source_points)} points on sphere (radius={radius})")
    print(f"  Target: transformed source with noise")
    print(f"  True transformation:")
    print(f"    Translation: ({true_tx:.2f}, {true_ty:.2f}, {true_tz:.2f})")
    print(f"    Rotation: Roll={np.degrees(true_roll):.1f}°, Pitch={np.degrees(true_pitch):.1f}°, Yaw={np.degrees(true_yaw):.1f}°")
    print(f"  Saved as: demo_3d_source.csv, demo_3d_target.csv")
    
    return true_tx, true_ty, true_tz, true_roll, true_pitch, true_yaw


def main():
    """Main function for 3D Gibbs sampler registration"""
    print("=" * 70)
    print("3D GIBBS SAMPLER POINT CLOUD REGISTRATION")
    print("=" * 70)
    
    # Skip demo data generation - using real Stanford Bunny data
    print("Using Stanford Bunny point cloud data...")
    
    # Initialize 3D Gibbs sampler with Stanford Bunny data
    gibbs_3d = GibbsSampler3DRegistration(
        "/Users/waelbenamara/Desktop/Research/RandomWalk/data/bunny_source.csv",
        "/Users/waelbenamara/Desktop/Research/RandomWalk/data/bunny_target.csv"
    )
    
    # Run Gibbs sampler
    success = gibbs_3d.run_gibbs_sampler(verbose=True)
    
    if success:
        # Plot diagnostics
        print("\nGenerating 3D trace plots...")
        gibbs_3d.plot_trace_plots()
        
        print("Generating 3D registration visualization...")
        gibbs_3d.visualize_3d_registration()
        
        # Load ground truth and evaluate results
        ground_truth = gibbs_3d.load_ground_truth("/Users/waelbenamara/Desktop/Research/RandomWalk/data/bunny_ground_truth.json")
        evaluation = gibbs_3d.evaluate_against_ground_truth(ground_truth)
        
        # Print evaluation results
        gibbs_3d.print_evaluation_results(ground_truth, evaluation)
        
        # Save results with evaluation
        gibbs_3d.save_results("gibbs_3d_registration", ground_truth, evaluation)
        
        return gibbs_3d
    else:
        print("3D Gibbs sampling failed")
        return None


if __name__ == "__main__":
    print("3D Gibbs Sampler for Bayesian Point Cloud Registration")
    print("Provides full posterior distributions over 3D transformation parameters\n")
    
    gibbs_3d_system = main()
    
    if gibbs_3d_system:
        print("\n" + "=" * 70)
        print("3D GIBBS SAMPLING COMPLETE!")
        print("=" * 70)
        print("Successfully sampled from the posterior distribution of")
        print("3D transformation parameters with full uncertainty quantification!")
        print("The circle has become a sphere!")
    else:
        print("3D Gibbs sampling failed")





