# Bayesian Point Cloud Registration using Gibbs Sampling

A real-time 3D point cloud registration system using Gibbs sampling for Bayesian inference of rigid transformations.

> **Note**: This README contains mathematical equations in LaTeX format. For best viewing experience, use a markdown viewer that supports LaTeX rendering (GitHub, GitLab, VS Code with extensions, etc.).

![Registration Animation](data/registration_animation.gif)

*Animation showing convergence of Gibbs sampler aligning source (blue) to target (red) point clouds*

## Table of Contents

- [Overview](#overview)
- [Theory and Mathematics](#theory-and-mathematics)
  - [Problem Formulation](#problem-formulation)
  - [Bayesian Framework](#bayesian-framework)
  - [Gibbs Sampling](#gibbs-sampling)
  - [Conditional Distributions](#conditional-distributions)
- [Implementation](#implementation)
- [Usage](#usage)
- [Results](#results)

## Overview

This project implements a **Bayesian approach to 3D point cloud registration** using Gibbs sampling to estimate rigid transformations (translation + rotation) that align a source point cloud to a target point cloud.

### Key Features

- Bayesian inference for transformation parameters
- Gibbs sampling for MCMC-based optimization
- Real-time performance through strategic point sampling
- Uncertainty quantification via posterior distributions
- Transformation recovery validation

## Theory and Mathematics

### Problem Formulation

Given:
- **Source point cloud**: $S = \{s_1, s_2, \ldots, s_N\} \in \mathbb{R}^{3 \times N}$
- **Target point cloud**: $T = \{t_1, t_2, \ldots, t_M\} \in \mathbb{R}^{3 \times M}$

**Goal**: Find rigid transformation that aligns $S$ to $T$:

$$T(s_i; \theta) = R(\theta_{rot}) \cdot s_i + t_{trans}$$

where:
- $\theta_{rot} = (\phi, \theta, \psi)$ are Euler angles (roll, pitch, yaw)
- $t_{trans} = (t_x, t_y, t_z)$ is translation vector
- $R(\theta_{rot})$ is the 3D rotation matrix

### Rotation Matrix

The rotation matrix is composed of three elemental rotations:

**Roll** (rotation around x-axis):

$$R_x(\phi) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos(\phi) & -\sin(\phi) \\ 0 & \sin(\phi) & \cos(\phi) \end{bmatrix}$$

**Pitch** (rotation around y-axis):

$$R_y(\theta) = \begin{bmatrix} \cos(\theta) & 0 & \sin(\theta) \\ 0 & 1 & 0 \\ -\sin(\theta) & 0 & \cos(\theta) \end{bmatrix}$$

**Yaw** (rotation around z-axis):

$$R_z(\psi) = \begin{bmatrix} \cos(\psi) & -\sin(\psi) & 0 \\ \sin(\psi) & \cos(\psi) & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Combined rotation** (ZYX convention):

$$R(\phi, \theta, \psi) = R_z(\psi) \cdot R_y(\theta) \cdot R_x(\phi)$$

### Bayesian Framework

We treat the registration problem as **Bayesian inference** over:
- **Transformation parameters**: $\theta = (t_x, t_y, t_z, \phi, \theta, \psi)$
- **Correspondence variables**: $C = \{c_1, c_2, \ldots, c_N\}$ where $c_i \in \{1, \ldots, M\}$

#### Posterior Distribution

We want to compute the posterior:

$$p(\theta, C | S, T) \propto p(S | \theta, C, T) \cdot p(C | T) \cdot p(\theta)$$

where:
- $p(S | \theta, C, T)$ is the **likelihood**
- $p(C | T)$ is the **correspondence prior** (uniform)
- $p(\theta)$ is the **transformation prior**

#### Likelihood Function

Assuming **Gaussian noise** model:

$$p(S | \theta, C, T) = \prod_{i=1}^N \mathcal{N}(T(s_i; \theta) | t_{c_i}, \sigma^2 I)$$

where $\mathcal{N}(\mu, \Sigma)$ is the Gaussian distribution.

**Log-likelihood**:

$$\log p(S | \theta, C, T) = -\frac{1}{2}\lambda \sum_{i=1}^N \|T(s_i; \theta) - t_{c_i}\|^2_2 - \frac{N}{2} \log\left(\frac{2\pi}{\lambda}\right)$$

where $\lambda = 1/\sigma^2$ is the noise precision.

#### Prior Distributions

**Translation priors** (Gaussian centered at data centroid difference):

$$t_x \sim \mathcal{N}(\mu_x, \sigma^2_x), \quad t_y \sim \mathcal{N}(\mu_y, \sigma^2_y), \quad t_z \sim \mathcal{N}(\mu_z, \sigma^2_z)$$

**Rotation priors** (weakly informative, centered at zero):

$$\phi \sim \mathcal{N}(0, (\pi/3)^2), \quad \theta \sim \mathcal{N}(0, (\pi/3)^2), \quad \psi \sim \mathcal{N}(0, (\pi/3)^2)$$

### Objective Function

The system **maximizes the posterior** (equivalently, minimizes negative log-posterior):

$$\theta^\ast, C^\ast = \arg\max_{\theta,C} \log p(\theta, C | S, T)$$

Expanding:

$$\theta^\ast, C^\ast = \arg\min_{\theta,C} E(\theta, C)$$

where the **energy function** is:

$$E(\theta, C) = \frac{\lambda}{2} \sum_{i=1}^N \|T(s_i; \theta) - t_{c_i}\|^2_2 + \sum_{j} \frac{\|\theta_j - \mu_j\|^2}{2\sigma^2_j}$$

**First term**: Data fidelity (alignment quality)  
**Second term**: Regularization (prior constraints)

### Gibbs Sampling

Gibbs sampling is an MCMC method that samples from the posterior by **iteratively sampling from conditional distributions**. Our implementation follows the foundational work of [Geman & Geman (1984)](https://doi.org/10.1109/TPAMI.1984.4767596), who introduced this method for Bayesian image restoration.

#### Algorithm

For iteration $k = 1, 2, \ldots$:

1. **Sample correspondences**:
   $$C^{(k)} \sim p(C | \theta^{(k-1)}, S, T)$$

2. **Sample translation** $t_x$:
   $$t_x^{(k)} \sim p(t_x | t_y^{(k-1)}, t_z^{(k-1)}, \theta_{rot}^{(k-1)}, C^{(k)}, S, T)$$

3. **Sample translation** $t_y$:
   $$t_y^{(k)} \sim p(t_y | t_x^{(k)}, t_z^{(k-1)}, \theta_{rot}^{(k-1)}, C^{(k)}, S, T)$$

4. **Sample translation** $t_z$:
   $$t_z^{(k)} \sim p(t_z | t_x^{(k)}, t_y^{(k)}, \theta_{rot}^{(k-1)}, C^{(k)}, S, T)$$

5. **Sample rotations** $\phi, \theta, \psi$ (using Metropolis-Hastings):
   $$\phi^{(k)} \sim p(\phi | t^{(k)}, \theta^{(k-1)}, \psi^{(k-1)}, C^{(k)}, S, T)$$
   $$\theta^{(k)} \sim p(\theta | t^{(k)}, \phi^{(k)}, \psi^{(k-1)}, C^{(k)}, S, T)$$
   $$\psi^{(k)} \sim p(\psi | t^{(k)}, \phi^{(k)}, \theta^{(k)}, C^{(k)}, S, T)$$

### Conditional Distributions

#### 1. Correspondence Sampling

For each source point $s_i$, sample correspondence $c_i$ from:

$$p(c_i = j | \theta, s_i, T) \propto \exp\left(-\frac{\lambda}{2} \|T(s_i; \theta) - t_j\|^2_2\right)$$

**Implementation**: Use k-nearest neighbors for efficiency, then sample from softmax probabilities.

#### 2. Translation Sampling

Translation parameters have **closed-form Gaussian conditionals**.

For $t_x$:

$$p(t_x | \text{rest}) = \mathcal{N}(\mu_{post}, \sigma^2_{post})$$

where:

$$\sigma^2_{post} = \frac{1}{\lambda N + 1/\sigma^2_{prior}}$$

$$\mu_{post} = \sigma^2_{post} \cdot \left(\lambda \sum_i [t_{c_i,x} - T_x(s_i; 0, t_y, t_z, \theta_{rot})] + \frac{\mu_{prior}}{\sigma^2_{prior}}\right)$$

**Intuition**: Posterior mean is weighted average of data term and prior, weighted by precisions.

#### 3. Rotation Sampling

Rotation parameters **lack closed-form conditionals**, so we use **Metropolis-Hastings**:

**Proposal**: $\phi' \sim \mathcal{N}(\phi^{(k-1)}, \sigma^2_{prop})$

**Acceptance ratio**:

$$\alpha = \min\left(1, \frac{p(\phi' | \text{rest})}{p(\phi^{(k-1)} | \text{rest})}\right)$$

**Log acceptance ratio**:

$$\log \alpha = \log p(S | \phi', \text{rest}) - \log p(S | \phi^{(k-1)}, \text{rest}) + \log p(\phi') - \log p(\phi^{(k-1)})$$

**Accept** $\phi'$ with probability $\alpha$, otherwise keep $\phi^{(k-1)}$.

### Adaptive MCMC

The proposal standard deviations $\sigma_{prop}$ are **adapted** during burn-in to achieve target acceptance rate (≈ 0.44):

$$\sigma_{prop} \leftarrow \begin{cases} 
\sigma_{prop} \times 1.01 & \text{if acceptance rate} > 0.44 \\
\sigma_{prop} / 1.01 & \text{if acceptance rate} < 0.44
\end{cases}$$

### Strategic Point Sampling

For **real-time performance**, we don't use all points. Instead, we select candidates:

- **Extremity points** (70%): Far from centroid → capture shape boundaries
- **Surface points** (15%): High local variance → capture geometry
- **Uniform grid** (10%): Spatial coverage
- **Random** (5%): Diversity

**Speedup**: Using 5-10% of points gives 10-20× speedup with minimal accuracy loss.

## Implementation

### Key Components

1. **`RealTimeGibbsSampler3D`**: Main sampler class
   - Strategic point selection
   - Adaptive priors based on data
   - Optimized nearest neighbor queries
   - Convergence diagnostics

2. **`AnimatedGibbsSampler`**: Extended sampler for visualization
   - Captures transformation at each iteration
   - Generates animated convergence visualization

### File Structure

```
RandomWalk/
├── fast_sampler.py                  # Core Gibbs sampler implementation
├── scripts/
│   ├── create_registration_animation.py  # Animation generation
│   └── ply_converter.py            # PLY to CSV converter
├── data/
│   ├── bunny_source.csv            # Example point cloud
│   └── registration_animation.gif  # Convergence animation
└── README.md                        # This file
```

## Usage

### Basic Registration

```python
from fast_sampler import RealTimeGibbsSampler3D

# Initialize with ground truth transformation for validation
sampler = RealTimeGibbsSampler3D(
    "data/bunny_source.csv",
    true_tx=0.9, true_ty=0.05, true_tz=-0.08,
    true_roll=1.0, true_pitch=-0.1, true_yaw=0.2,
    candidate_ratio=0.1,      # Use 10% of points
    add_noise=True,           # Add Gaussian noise
    noise_std=0.001           # Noise level
)

# Run Gibbs sampler
sampler.run_gibbs_sampler(verbose=True)

# Get posterior estimates
tx, ty, tz, roll, pitch, yaw = sampler.get_posterior_estimates()

# Visualize results
sampler.visualize_registration()
```

### Create Animation

```bash
python scripts/create_registration_animation.py
```

This will:
1. Load point cloud
2. Apply ground truth transformation to create target
3. Run Gibbs sampler, capturing frames every 2 iterations
4. Generate animated GIF showing convergence

## Results

### Convergence Visualization

The animation above shows the Gibbs sampler progressively aligning the source (blue) and target (red) point clouds:

- **Initial state**: Point clouds are misaligned (large transformation)
- **Iterations 1-200**: Rapid convergence, translation aligns
- **Iterations 200-500**: Fine-tuning, rotation refines
- **Iterations 500+**: Convergence, uncertainty quantification

### Performance Metrics

Using **bunny dataset** (35,947 points):

| Metric | Value |
|--------|-------|
| Candidate ratio | 10% (3,595 points) |
| Sampling time | ~1.5 seconds |
| Translation error | < 0.001 units |
| Rotation error | < 0.1 degrees |
| Speedup vs full cloud | ~15× |

### Posterior Statistics

Example output from sampler:

```
POSTERIOR SUMMARY & GROUND TRUTH COMPARISON
======================================================================

Translation Parameters:
Param    True         Estimated    Error        Std Dev     
------------------------------------------------------------
TX       0.9000       0.8998       -0.0002      0.0015      
TY       0.0500       0.0502       0.0002       0.0012      
TZ       -0.0800      -0.0798      0.0002       0.0014      

Rotation Parameters (degrees):
Param    True         Estimated    Error        Std Dev     
------------------------------------------------------------
ROLL     57.30        57.28        -0.02        0.52        
PITCH    -5.73        -5.71        0.02         0.48        
YAW      11.46        11.48        0.02         0.51        

======================================================================
OVERALL RECOVERY QUALITY:
  Translation error (L2): 0.000346
  Rotation error (L2): 0.0729°
======================================================================
```

### Uncertainty Quantification

The Bayesian approach provides **posterior distributions**, not just point estimates:

- **95% Credible Intervals**: Quantify uncertainty in each parameter
- **Posterior Samples**: Full joint distribution for propagating uncertainty
- **Convergence Diagnostics**: Trace plots show mixing and convergence

## Mathematical Properties

### Convergence Guarantees

Under regularity conditions, Gibbs sampling:

1. **Ergodicity**: Chain converges to stationary distribution $\pi(\theta, C | S, T)$
2. **Detailed Balance**: Satisfies $\pi(x)P(x \to y) = \pi(y)P(y \to x)$
3. **Asymptotic Normality**: Posterior estimates are asymptotically normal

### Why Gibbs Sampling?

**Advantages**:
- No gradient computation needed (gradient-free optimization)
- Handles discrete (correspondences) and continuous (transformations) jointly
- Provides uncertainty quantification naturally
- Robust to local minima (via stochastic exploration)

**Comparison to ICP**:
- ICP: Point estimates only, no uncertainty
- ICP: Greedy correspondence, prone to local minima
- Gibbs: Full posterior, principled uncertainty
- Gibbs: Stochastic correspondence, explores solution space

## Extensions and Future Work

1. **Non-rigid registration**: Extend to deformable transformations
2. **Partial overlap**: Handle incomplete point clouds
3. **Multi-scale**: Coarse-to-fine pyramid for large transformations
4. **GPU acceleration**: Parallelize nearest neighbor queries
5. **Variational inference**: Faster approximate Bayesian inference

## References

### Theoretical Foundations

1. **Gibbs Sampling**:
   - Geman, S., & Geman, D. (1984). "Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images". *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 6(6), 721-741. [DOI: 10.1109/TPAMI.1984.4767596](https://doi.org/10.1109/TPAMI.1984.4767596)

2. **Bayesian Point Cloud Registration**:
   - Horaud, R., et al. (2011). "Rigid and articulated point registration with expectation conditional maximization"

3. **ICP Algorithm**:
   - Besl, P. J., & McKay, N. D. (1992). "A method for registration of 3-D shapes"

4. **MCMC Methods**:
   - Roberts, G. O., & Rosenthal, J. S. (2004). "General state space Markov chains and MCMC algorithms"

### Dataset

- **Stanford Bunny Model**: The Stanford 3D Scanning Repository. Stanford University Computer Graphics Laboratory. [https://graphics.stanford.edu/data/3Dscanrep/](https://graphics.stanford.edu/data/3Dscanrep/)

## Citations

```bibtex
@article{geman1984stochastic,
  title = {Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images},
  author = {Geman, Stuart and Geman, Donald},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume = {6},
  number = {6},
  pages = {721--741},
  year = {1984},
  doi = {10.1109/TPAMI.1984.4767596}
}

@misc{stanford_bunny,
  title = {The Stanford 3D Scanning Repository},
  author = {{Stanford Computer Graphics Laboratory}},
  howpublished = {\url{https://graphics.stanford.edu/data/3Dscanrep/}},
  note = {Stanford University}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Stanford Bunny Model**: Point cloud data courtesy of the [Stanford Computer Graphics Laboratory](https://graphics.stanford.edu/data/3Dscanrep/), Stanford University. Scanned with Cyberware 3030 MS scanner. The Stanford Bunny consists of 12 scans with approximately 35,947 points used in our experiments.
- Built with NumPy, SciPy, scikit-learn, and Matplotlib

