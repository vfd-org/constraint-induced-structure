"""
Dimensionality and embedding analysis for CISE.

Provides PCA analysis, intrinsic dimension estimation, and clustering tendency.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_pca(
    samples: np.ndarray,
    n_components: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, PCA]:
    """Compute PCA projection of samples.

    Args:
        samples: Array of samples (n_samples, dim) or (n_samples, k, k).
        n_components: Number of components (default: min(n_samples, n_features)).
        weights: Optional importance weights (not used in sklearn PCA).

    Returns:
        Tuple of (projected_samples, pca_model).
    """
    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)

    if n_components is None:
        n_components = min(n_samples, samples_flat.shape[1])

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(samples_flat)

    return projected, pca


def compute_pca_explained_variance(
    samples: np.ndarray,
    n_components: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute PCA explained variance curve.

    Args:
        samples: Array of samples.
        n_components: Number of components to compute.

    Returns:
        Dictionary with explained variance information.
    """
    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)

    if n_components is None:
        n_components = min(n_samples, samples_flat.shape[1], 50)

    pca = PCA(n_components=n_components)
    pca.fit(samples_flat)

    explained_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained_ratio)

    # Find number of components for various thresholds
    thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    n_components_for = {}
    for t in thresholds:
        idx = np.searchsorted(cumulative, t)
        n_components_for[f"n_for_{int(t*100)}pct"] = int(idx + 1)

    return {
        "explained_variance_ratio": explained_ratio.tolist(),
        "cumulative_variance": cumulative.tolist(),
        "n_components": int(n_components),
        **n_components_for,
    }


def compute_intrinsic_dimension(
    samples: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute intrinsic dimension proxy via participation ratio.

    Participation ratio: PR = (sum lambda_i)^2 / sum(lambda_i^2)
    where lambda_i are eigenvalues (or explained variances).

    Args:
        samples: Array of samples.
        weights: Optional importance weights.

    Returns:
        Dictionary with intrinsic dimension estimates.
    """
    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)

    n_components = min(n_samples - 1, samples_flat.shape[1])
    if n_components < 2:
        return {"participation_ratio": 1.0, "n_components_used": 1}

    pca = PCA(n_components=n_components)
    pca.fit(samples_flat)

    # Explained variances (eigenvalues)
    lambdas = pca.explained_variance_

    # Participation ratio
    sum_lambdas = np.sum(lambdas)
    sum_lambdas_sq = np.sum(lambdas ** 2)

    if sum_lambdas_sq > 0:
        pr = (sum_lambdas ** 2) / sum_lambdas_sq
    else:
        pr = 1.0

    return {
        "participation_ratio": float(pr),
        "n_components_used": int(n_components),
    }


def compute_clustering_tendency(
    samples: np.ndarray,
    k_range: List[int] = None,
    weights: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute clustering tendency via silhouette scores.

    Args:
        samples: Array of samples.
        k_range: Range of k values to try (default [2, 3, 4, 5, 6]).
        weights: Optional importance weights (for weighted resampling).
        random_state: Random state for K-means.

    Returns:
        Dictionary with silhouette scores and best k.
    """
    if k_range is None:
        k_range = [2, 3, 4, 5, 6]

    n_samples = samples.shape[0]
    samples_flat = samples.reshape(n_samples, -1)

    # If too few samples, skip clustering
    if n_samples < max(k_range) + 1:
        return {
            "silhouette_scores": {},
            "best_k": None,
            "best_silhouette": None,
            "note": "Too few samples for clustering",
        }

    # If using weights, resample to create weighted ensemble
    if weights is not None:
        rng = np.random.default_rng(random_state)
        weights_norm = weights / np.sum(weights)
        indices = rng.choice(n_samples, size=n_samples, replace=True, p=weights_norm)
        samples_flat = samples_flat[indices]

    silhouette_scores = {}
    for k in k_range:
        if k >= n_samples:
            continue
        try:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(samples_flat)
            score = silhouette_score(samples_flat, labels)
            silhouette_scores[k] = float(score)
        except Exception:
            silhouette_scores[k] = None

    # Find best k
    valid_scores = {k: v for k, v in silhouette_scores.items() if v is not None}
    if valid_scores:
        best_k = max(valid_scores, key=valid_scores.get)
        best_silhouette = valid_scores[best_k]
    else:
        best_k = None
        best_silhouette = None

    return {
        "silhouette_scores": silhouette_scores,
        "best_k": best_k,
        "best_silhouette": best_silhouette,
    }
