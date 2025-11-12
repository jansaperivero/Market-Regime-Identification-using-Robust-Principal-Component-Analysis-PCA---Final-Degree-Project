"""
Simple Market Model
R(t,i) = m(i) + a(i)·w(t) + z(t,i)

Where:
- m(i): Expected return of asset i
- a(i): Weight of asset i  
- w(t): Market trend at time t
- z(t,i): Noise
"""

import numpy as np
import matplotlib.pyplot as plt

def perform_pca(R, standardize=True):
    """
    Perform PCA on returns matrix R (shape: periods x assets).
    
    Steps:
    1) Center each asset's returns (subtract mean)
    2) Optionally standardize by per-asset std (z-score)
    3) Compute SVD of the standardized matrix
       R_std = U S V^T
       - Columns of V (rows of V^T) are principal axes (loadings)
       - Scores (time-series of PCs) = R_std @ V
    4) Explained variance from singular values
    """
    T, N = R.shape
    # 1) center
    means = R.mean(axis=0)
    centered = R - means
    # 2) standardize
    if standardize:
        stds = R.std(axis=0, ddof=1)
        # avoid divide by zero
        stds_safe = np.where(stds == 0, 1.0, stds)
        X = centered / stds_safe
    else:
        stds_safe = np.ones_like(means)
        X = centered
    # 3) SVD
    # Note: Use economy SVD for efficiency
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # 4) Explained variance
    # For standardized data, covariance of X is (1/(T-1)) X^T X
    # Eigenvalues of covariance are (S^2)/(T-1)
    eigvals = (S ** 2) / (T - 1)
    total_var = eigvals.sum()
    explained_variance_ratio = eigvals / total_var if total_var > 0 else np.zeros_like(eigvals)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    # Scores (PC time series) and loadings (asset contributions)
    scores = U * S  # same as X @ V
    loadings = Vt.T  # shape: assets x components
    return {
        'means': means,
        'stds': stds_safe,
        'scores': scores,
        'loadings': loadings,
        'singular_values': S,
        'eigenvalues': eigvals,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
    }

def plot_pca_results(pca_result):
    """Plot scree and first PC loadings and scores."""
    evr = pca_result['explained_variance_ratio']
    cum = pca_result['cumulative_variance']
    loadings = pca_result['loadings']
    scores = pca_result['scores']
    n_components = min(5, len(evr))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Scree plot
    axes[0, 0].bar(range(1, n_components + 1), evr[:n_components], alpha=0.8)
    axes[0, 0].plot(range(1, n_components + 1), cum[:n_components], 'ro--', label='Cumulative')
    axes[0, 0].set_title('PCA Explained Variance (Scree)')
    axes[0, 0].set_xlabel('Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    # First PC loadings (assets contribution)
    axes[0, 1].bar(range(loadings.shape[0]), loadings[:, 0], alpha=0.8)
    axes[0, 1].set_title('PC1 Loadings (Asset Contributions)')
    axes[0, 1].set_xlabel('Asset')
    axes[0, 1].set_ylabel('Loading')
    axes[0, 1].grid(True, alpha=0.3)
    # First PC scores over time (common factor series)
    axes[1, 0].plot(scores[:, 0], color='purple', linewidth=1.5)
    axes[1, 0].set_title('PC1 Scores (Common Factor Time Series)')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].grid(True, alpha=0.3)
    # Second PC scores (if available)
    if scores.shape[1] > 1:
        axes[1, 1].plot(scores[:, 1], color='teal', linewidth=1.5)
        axes[1, 1].set_title('PC2 Scores')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    plt.tight_layout()
    plt.show()

def generate_returns(n_assets=3, n_periods=100, seed=42):
    """
    Generate asset returns using the model:
    R(t,i) = m(i) + a(i)·w(t) + z(t,i)
    """
    np.random.seed(seed)
    
    # Parameters
    m = np.random.normal(0.001, 0.0005, n_assets)  # Expected returns
    a = np.random.uniform(0.5, 1.5, n_assets)      # Asset weights
    w = np.random.normal(0, 0.02, n_periods)       # Market trend
    z = np.random.normal(0, 0.01, (n_periods, n_assets))  # Noise
    
    # Generate returns matrix
    R = np.zeros((n_periods, n_assets))
    for t in range(n_periods):
        for i in range(n_assets):
            R[t, i] = m[i] + a[i] * w[t] + z[t, i]
    
    return R, m, a, w, z

def plot_results(R, m, a, w, z):
    """Plot the generated data"""
    n_periods, n_assets = R.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot returns
    for i in range(n_assets):
        axes[0, 0].plot(R[:, i], label=f'Asset {i+1}', alpha=0.7)
    axes[0, 0].set_title('Asset Returns R(t,i)')
    axes[0, 0].set_xlabel('Time t')
    axes[0, 0].set_ylabel('Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot market trend
    axes[0, 1].plot(w, color='red', linewidth=2)
    axes[0, 1].set_title('Market Trend w(t)')
    axes[0, 1].set_xlabel('Time t')
    axes[0, 1].set_ylabel('Market Factor')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot asset weights
    axes[1, 0].bar(range(n_assets), a, color=['blue', 'green', 'red'][:n_assets])
    axes[1, 0].set_title('Asset Weights a(i)')
    axes[1, 0].set_xlabel('Asset i')
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_xticks(range(n_assets))
    axes[1, 0].set_xticklabels([f'Asset {i+1}' for i in range(n_assets)])
    
    # Plot expected returns
    axes[1, 1].bar(range(n_assets), m, color=['blue', 'green', 'red'][:n_assets])
    axes[1, 1].set_title('Expected Returns m(i)')
    axes[1, 1].set_xlabel('Asset i')
    axes[1, 1].set_ylabel('Expected Return')
    axes[1, 1].set_xticks(range(n_assets))
    axes[1, 1].set_xticklabels([f'Asset {i+1}' for i in range(n_assets)])
    
    plt.tight_layout()
    plt.show()

def print_model_details(R, m, a, w, z):
    """Print model details"""
    n_periods, n_assets = R.shape
    
    print("=" * 50)
    print("SIMPLE MARKET MODEL")
    print("=" * 50)
    print(f"Model: R(t,i) = m(i) + a(i)·w(t) + z(t,i)")
    print(f"Matrix dimensions: {n_periods} periods × {n_assets} assets")
    print()
    
    print("Parameters:")
    for i in range(n_assets):
        print(f"  Asset {i+1}: m({i+1}) = {m[i]:.4f}, a({i+1}) = {a[i]:.3f}")
    print()
    
    print("Sample calculations (first 3 periods):")
    for t in range(3):
        print(f"  Period {t+1}:")
        for i in range(n_assets):
            r_ti = m[i] + a[i] * w[t] + z[t, i]
            print(f"    R({t+1},{i+1}) = {m[i]:.4f} + {a[i]:.3f}×{w[t]:.4f} + {z[t, i]:.4f} = {r_ti:.4f}")
    print()
    
    print("Matrix properties:")
    print(f"  Shape: {R.shape}")
    print(f"  Rank: {np.linalg.matrix_rank(R)}")
    print(f"  Mean returns: {np.mean(R, axis=0)}")
    print(f"  Std returns: {np.std(R, axis=0)}")

if __name__ == "__main__":
    # Generate data
    R, m, a, w, z = generate_returns(n_assets=3, n_periods=100, seed=42)
    
    # Print details
    print_model_details(R, m, a, w, z)
    
    # PCA: identify common factors and variance explained
    pca = perform_pca(R, standardize=True)
    print("PCA SUMMARY")
    print("-" * 50)
    print(f"Explained variance ratio: {np.round(pca['explained_variance_ratio'], 4)}")
    print(f"Cumulative variance: {np.round(pca['cumulative_variance'], 4)}")
    k80 = np.argmax(pca['cumulative_variance'] >= 0.80) + 1
    print(f"Components to reach 80% variance: {k80}")
    # Optional visualization of PCA
    plot_pca_results(pca)
    
    # Plot results
    plot_results(R, m, a, w, z)
