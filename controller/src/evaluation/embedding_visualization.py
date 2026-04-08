"""
Visualization utilities for scenario embeddings.

Provides functions to visualize and analyze scenario encoder embeddings:
- 2D projection (t-SNE/PCA)
- Embedding distances and similarities
- Correlation with structural parameters
- Evolution during training
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr


def plot_embedding_tsne(embeddings, structural_params, scenario_indices, save_path,
                        param_names=None, perplexity=30):
    """
    Plot t-SNE 2D projection of embeddings colored by structural parameters.

    Args:
        embeddings: Array of shape (n_scenarios, embedding_dim)
        structural_params: Array of shape (n_scenarios, n_params)
        scenario_indices: Array of scenario indices
        save_path: Path to save the figure
        param_names: List of parameter names (default: ['Param 0', 'Param 1', ...])
        perplexity: t-SNE perplexity parameter
    """
    n_scenarios, n_params = structural_params.shape

    if param_names is None:
        param_names = [f'Param {i}' for i in range(n_params)]

    # Compute t-SNE projection
    if n_scenarios < perplexity:
        perplexity = max(5, n_scenarios // 2)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create figure with subplots for each structural parameter (larger for better visibility)
    n_cols = min(3, n_params + 1)  # Max 3 columns
    n_rows = (n_params + 1 + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))  # Increased from (5, 4)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    axes = axes.flatten()

    # Plot 1: All scenarios
    scatter = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                             c=scenario_indices, cmap='viridis', s=30, alpha=0.6)
    axes[0].set_title('t-SNE Projection\n(colored by scenario index)', fontsize=10)
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter, ax=axes[0], label='Scenario Index')

    # Plot 2-N: Colored by each structural parameter
    for i in range(n_params):
        ax_idx = i + 1
        param_values = structural_params[:, i]
        scatter = axes[ax_idx].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                      c=param_values, cmap='coolwarm', s=30, alpha=0.6)
        axes[ax_idx].set_title(f't-SNE Projection\n(colored by {param_names[i]})', fontsize=10)
        axes[ax_idx].set_xlabel('t-SNE Dimension 1')
        axes[ax_idx].set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter, ax=axes[ax_idx], label=param_names[i])

    # Hide unused subplots
    for i in range(n_params + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Increased DPI from 150 to 200
    plt.close()
    print(f"  ✓ t-SNE plot saved: {save_path}")


def plot_embedding_pca(embeddings, structural_params, scenario_indices, save_path,
                       param_names=None):
    """
    Plot PCA 2D projection of embeddings colored by structural parameters.

    Args:
        embeddings: Array of shape (n_scenarios, embedding_dim)
        structural_params: Array of shape (n_scenarios, n_params)
        scenario_indices: Array of scenario indices
        save_path: Path to save the figure
        param_names: List of parameter names
    """
    n_scenarios, n_params = structural_params.shape

    if param_names is None:
        param_names = [f'Param {i}' for i in range(n_params)]

    # Compute PCA projection
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    explained_var = pca.explained_variance_ratio_

    # Create figure with subplots (larger for better visibility)
    n_cols = min(3, n_params + 1)
    n_rows = (n_params + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))  # Increased from (5, 4)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    axes = axes.flatten()

    # Plot 1: All scenarios
    scatter = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                             c=scenario_indices, cmap='viridis', s=30, alpha=0.6)
    axes[0].set_title('PCA Projection\n(colored by scenario index)', fontsize=10)
    axes[0].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
    plt.colorbar(scatter, ax=axes[0], label='Scenario Index')

    # Plot 2-N: Colored by each structural parameter
    for i in range(n_params):
        ax_idx = i + 1
        param_values = structural_params[:, i]
        scatter = axes[ax_idx].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                      c=param_values, cmap='coolwarm', s=30, alpha=0.6)
        axes[ax_idx].set_title(f'PCA Projection\n(colored by {param_names[i]})', fontsize=10)
        axes[ax_idx].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
        axes[ax_idx].set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
        plt.colorbar(scatter, ax=axes[ax_idx], label=param_names[i])

    # Hide unused subplots
    for i in range(n_params + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Increased DPI from 150 to 200
    plt.close()
    print(f"  ✓ PCA plot saved: {save_path}")


def plot_embedding_distances(embeddings, structural_params, scenario_indices, save_path,
                             param_names=None, n_samples=50):
    """
    Plot embedding distances vs structural parameter distances.

    Shows correlation between embedding similarity and operational condition similarity.

    Args:
        embeddings: Array of shape (n_scenarios, embedding_dim)
        structural_params: Array of shape (n_scenarios, n_params)
        scenario_indices: Array of scenario indices
        save_path: Path to save the figure
        param_names: List of parameter names
        n_samples: Number of scenario pairs to plot (for performance)
    """
    n_scenarios, n_params = structural_params.shape

    if param_names is None:
        param_names = [f'Param {i}' for i in range(n_params)]

    # Compute pairwise embedding distances
    embedding_dists = pdist(embeddings, metric='euclidean')

    # Create figure (larger for better visibility)
    fig, axes = plt.subplots(1, n_params + 1, figsize=(6*(n_params+1), 5))  # Increased from (5, 4)
    if n_params == 0:
        axes = [axes]

    # Plot 1: Overall structural distance (L2 norm across all params)
    struct_dists = pdist(structural_params, metric='euclidean')

    # Sample for visualization (full distance matrices can be huge)
    n_total_pairs = len(embedding_dists)
    if n_total_pairs > n_samples:
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(n_total_pairs, n_samples, replace=False)
        struct_dists_sample = struct_dists[sample_indices]
        embedding_dists_sample = embedding_dists[sample_indices]
    else:
        struct_dists_sample = struct_dists
        embedding_dists_sample = embedding_dists

    # Correlation
    corr, p_value = pearsonr(struct_dists_sample, embedding_dists_sample)

    axes[0].scatter(struct_dists_sample, embedding_dists_sample, alpha=0.5, s=20)
    axes[0].set_xlabel('Structural Distance (L2)')
    axes[0].set_ylabel('Embedding Distance (L2)')
    axes[0].set_title(f'Overall Correlation\n(r={corr:.3f}, p={p_value:.2e})', fontsize=10)
    axes[0].grid(alpha=0.3)

    # Plot 2-N: Per-parameter distances
    for i in range(n_params):
        param_dists = pdist(structural_params[:, i:i+1], metric='euclidean')

        if n_total_pairs > n_samples:
            param_dists_sample = param_dists[sample_indices]
        else:
            param_dists_sample = param_dists

        corr_param, p_val_param = pearsonr(param_dists_sample, embedding_dists_sample)

        axes[i+1].scatter(param_dists_sample, embedding_dists_sample, alpha=0.5, s=20)
        axes[i+1].set_xlabel(f'{param_names[i]} Distance')
        axes[i+1].set_ylabel('Embedding Distance (L2)')
        axes[i+1].set_title(f'{param_names[i]} Correlation\n(r={corr_param:.3f}, p={p_val_param:.2e})', fontsize=10)
        axes[i+1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Increased DPI from 150 to 200
    plt.close()
    print(f"  ✓ Distance correlation plot saved: {save_path}")


def plot_embedding_correlation_heatmap(embeddings, structural_params, save_path,
                                       param_names=None):
    """
    Plot correlation heatmap between embedding dimensions and structural parameters.

    Shows which embedding dimensions capture which structural information.

    Args:
        embeddings: Array of shape (n_scenarios, embedding_dim)
        structural_params: Array of shape (n_scenarios, n_params)
        save_path: Path to save the figure
        param_names: List of parameter names
    """
    n_scenarios, embedding_dim = embeddings.shape
    n_params = structural_params.shape[1]

    if param_names is None:
        param_names = [f'Param {i}' for i in range(n_params)]

    # Compute correlation matrix
    corr_matrix = np.zeros((embedding_dim, n_params))

    for i in range(embedding_dim):
        for j in range(n_params):
            corr, _ = pearsonr(embeddings[:, i], structural_params[:, j])
            corr_matrix[i, j] = corr

    # Create heatmap (larger for better visibility)
    fig, ax = plt.subplots(figsize=(max(8, n_params*2), max(6, embedding_dim*0.4)))  # Increased dimensions

    sns.heatmap(corr_matrix,
                xticklabels=param_names,
                yticklabels=[f'Emb {i}' for i in range(embedding_dim)],
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot=True if embedding_dim <= 32 else False,
                fmt='.2f',
                cbar_kws={'label': 'Pearson Correlation'},
                ax=ax)

    ax.set_title('Embedding-Parameter Correlations', fontsize=12, pad=15)
    ax.set_xlabel('Structural Parameters', fontsize=10)
    ax.set_ylabel('Embedding Dimensions', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Increased DPI from 150 to 200
    plt.close()
    print(f"  ✓ Correlation heatmap saved: {save_path}")


def plot_embedding_evolution(embedding_history, scenario_indices_sample, save_path,
                             epochs_to_show=None, n_scenarios_show=10):
    """
    Plot how embeddings evolve during training for selected scenarios.

    Args:
        embedding_history: Dict with keys = epoch numbers, values = embeddings array
        scenario_indices_sample: Which scenarios to track
        save_path: Path to save the figure
        epochs_to_show: List of specific epochs to visualize (default: evenly spaced)
        n_scenarios_show: Number of scenarios to track
    """
    if len(embedding_history) == 0:
        print("  ⚠ No embedding history to plot")
        return

    epochs = sorted(embedding_history.keys())

    if epochs_to_show is None:
        # Show evenly spaced epochs
        n_epochs_show = min(6, len(epochs))
        step = max(1, len(epochs) // n_epochs_show)
        epochs_to_show = epochs[::step]

    # Select scenarios to track
    first_epoch_embeddings = embedding_history[epochs[0]]
    n_scenarios = first_epoch_embeddings.shape[0]

    if scenario_indices_sample is None or len(scenario_indices_sample) == 0:
        # Sample evenly across scenario range
        scenario_indices_sample = np.linspace(0, n_scenarios-1, n_scenarios_show, dtype=int)
    else:
        scenario_indices_sample = scenario_indices_sample[:n_scenarios_show]

    # Collect embeddings for selected scenarios
    embedding_trajectories = {scenario_idx: [] for scenario_idx in scenario_indices_sample}

    for epoch in epochs:
        embeddings = embedding_history[epoch]
        for scenario_idx in scenario_indices_sample:
            embedding_trajectories[scenario_idx].append(embeddings[scenario_idx])

    # Project all embeddings to 2D using PCA (fit on all epochs combined)
    all_embeddings = np.concatenate([embedding_history[e] for e in epochs], axis=0)
    pca = PCA(n_components=2, random_state=42)
    all_embeddings_2d = pca.fit_transform(all_embeddings)

    # Split back by epoch
    n_per_epoch = n_scenarios
    embeddings_2d_by_epoch = {}
    start_idx = 0
    for epoch in epochs:
        embeddings_2d_by_epoch[epoch] = all_embeddings_2d[start_idx:start_idx+n_per_epoch]
        start_idx += n_per_epoch

    # Create figure (larger for better visibility)
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased from (10, 8)

    # Plot trajectories for each selected scenario
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_indices_sample)))

    for i, scenario_idx in enumerate(scenario_indices_sample):
        trajectory_2d = [embeddings_2d_by_epoch[e][scenario_idx] for e in epochs]
        trajectory_2d = np.array(trajectory_2d)

        # Plot trajectory line
        ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1],
               'o-', color=colors[i], alpha=0.6, linewidth=1.5,
               markersize=4, label=f'Scenario {scenario_idx}')

        # Mark start and end
        ax.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1],
                  marker='s', s=80, color=colors[i], edgecolors='black', linewidths=1.5,
                  zorder=10)  # Square for start
        ax.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1],
                  marker='*', s=120, color=colors[i], edgecolors='black', linewidths=1.5,
                  zorder=10)  # Star for end

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'Embedding Evolution During Training\n(Epochs: {epochs[0]} → {epochs[-1]})')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Add legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=8, label='Epoch 1', markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
               markersize=10, label=f'Epoch {epochs[-1]}', markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Increased DPI from 150 to 200
    plt.close()
    print(f"  ✓ Embedding evolution plot saved: {save_path}")


def generate_all_embedding_plots(embeddings, structural_params, scenario_indices,
                                 checkpoint_dir, embedding_history=None, param_names=None):
    """
    Generate all embedding visualization plots.

    Args:
        embeddings: Final embeddings array (n_scenarios, embedding_dim)
        structural_params: Structural parameters array (n_scenarios, n_params)
        scenario_indices: Array of scenario indices
        checkpoint_dir: Directory to save plots
        embedding_history: Dict of embeddings by epoch (optional)
        param_names: List of structural parameter names

    Returns:
        Dict mapping plot names to file paths
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 Generating embedding visualization plots...")

    plot_paths = {}

    # 1. t-SNE projection
    tsne_path = checkpoint_dir / 'embedding_tsne.png'
    plot_embedding_tsne(embeddings, structural_params, scenario_indices, tsne_path, param_names)
    plot_paths['tsne'] = tsne_path

    # 2. PCA projection
    pca_path = checkpoint_dir / 'embedding_pca.png'
    plot_embedding_pca(embeddings, structural_params, scenario_indices, pca_path, param_names)
    plot_paths['pca'] = pca_path

    # 3. Distance correlation
    dist_path = checkpoint_dir / 'embedding_distances.png'
    plot_embedding_distances(embeddings, structural_params, scenario_indices, dist_path, param_names)
    plot_paths['distances'] = dist_path

    # 4. Correlation heatmap
    corr_path = checkpoint_dir / 'embedding_correlations.png'
    plot_embedding_correlation_heatmap(embeddings, structural_params, corr_path, param_names)
    plot_paths['correlations'] = corr_path

    # 5. Evolution (if history available)
    if embedding_history is not None and len(embedding_history) > 1:
        evol_path = checkpoint_dir / 'embedding_evolution.png'
        plot_embedding_evolution(embedding_history, scenario_indices[:10], evol_path)
        plot_paths['evolution'] = evol_path

    print(f"✓ All embedding plots generated in {checkpoint_dir}")

    return plot_paths
