"""
Analysis module for CISE.

Provides metrics computation, dimensionality analysis, and plotting functions.
"""

from cise.analysis.metrics import (
    compute_metrics,
    compute_norm_stats,
    compute_energy_stats,
    compute_compressibility,
    compute_gini_coefficient,
    compute_matrix_metrics,
)
from cise.analysis.embeddings import (
    compute_pca,
    compute_pca_explained_variance,
    compute_intrinsic_dimension,
    compute_clustering_tendency,
)
from cise.analysis.plots import (
    plot_energy_histogram,
    plot_norm_distributions,
    plot_pca_variance,
    plot_pca_scatter,
    plot_singular_value_spectra,
    plot_hierarchy,
    create_all_plots,
)
from cise.analysis.release_plots import (
    create_release_figures,
    plot_ess_vs_beta,
)
from cise.analysis.controls import (
    create_norm_matched_baseline,
    create_control_figures,
    compute_control_metrics,
)

__all__ = [
    # Metrics
    "compute_metrics",
    "compute_norm_stats",
    "compute_energy_stats",
    "compute_compressibility",
    "compute_gini_coefficient",
    "compute_matrix_metrics",
    # Embeddings
    "compute_pca",
    "compute_pca_explained_variance",
    "compute_intrinsic_dimension",
    "compute_clustering_tendency",
    # Plots
    "plot_energy_histogram",
    "plot_norm_distributions",
    "plot_pca_variance",
    "plot_pca_scatter",
    "plot_singular_value_spectra",
    "plot_hierarchy",
    "create_all_plots",
    # Release plots
    "create_release_figures",
    "plot_ess_vs_beta",
    # Controls
    "create_norm_matched_baseline",
    "create_control_figures",
    "compute_control_metrics",
]
