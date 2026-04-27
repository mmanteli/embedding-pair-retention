import numpy as np
from dataclasses import dataclass
from typing import Iterable, Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

def _as_2d_float(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")
    return a


def pairwise_distances(X, metric="cosine"):
    """
    Make a pairwise distance matrix for input X
    """
    X = _as_2d_float(X)
    return squareform(pdist(X, metric=metric))

def knn_indices_from_distance_matrix(D, k:int):
    """
    Return indices of k nearest neighbors for each row i, excluding self.
    D: NxN pairwise distances.
    Output: Nxk integer array of indices.
    """
    D = np.asarray(D)
    N = D.shape[0]
    if D.shape != (N, N):
        raise ValueError("D must be sqaure")
    if not (1 <= k <= N - 1):
        raise ValueError(f"k value must be in [1, {N-1}], got {k}")
    idx = np.argsort(D, axis=1)
    # exclude self -> d(self, self) == 0, always the first index
    idx = idx[:, 1:k+1]
    return idx

def _jaccard_overlap(a, b):
    """Jaccard overlap for two 1D integer arrays (as sets)."""
    sa = set(map(int, a))
    sb = set(map(int, b))
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 1.0

def knn_overlap_score(X, Y, k=10, metric="cosine", mode= "jaccard"):
    """
    Calculate the average neighborhood preservation score, 0 to 1
      - "recall": |N_k^X(i) ∩ N_k^Y(i)| / k averaged over i
      - "jaccard": Jaccard(N_k^X(i), N_k^Y(i)) averaged over i
    """
    assert mode in ["recall", "jaccard"], f"Mode needs to be recall or jaccard, now {mode}"
    assert X.shape[0] == Y.shape[0], f"X and Y must have the same number of instances, X.shape={X.shape}, Y.shape={Y.shape}"

    DX = pairwise_distances(X, metric=metric)
    DY = pairwise_distances(Y, metric=metric)

    NX = knn_indices_from_distance_matrix(DX, k)
    NY = knn_indices_from_distance_matrix(DY, k)

    if mode == "recall":
        scores = []
        for i in range(X.shape[0]):
            scores.append(len(set(NX[i]) & set(NY[i])) / k)
        return float(np.mean(scores))

    else:
        return float(np.mean([_jaccard_overlap(NX[i], NY[i]) for i in range(X.shape[0])]))


def knn_overlap_scores_multi_k(X,Y, ks=(5, 10, 20, 50),metric="cosine",mode="jaccard"):
    """Compute kNN overlap score for multiple k  values."""

    N = X.shape[0]
    out = {}
    for k in ks:
        k = int(k)
        if k < 1 or k >= N:
            print(f"Badly given k value {k} (1...{N}); skipping calculations")
            continue
        out[k] = knn_overlap_score(X, Y, k=k, metric=metric, mode=mode)
    return out

def distance_correlation_scores(X, Y, metric="cosine", rank=False):
    """
    Calculate correlation between all pairwise distances.
    rank=False => Pearson correlation
    rank=True  => Spearman correlation
    """

    dx = pdist(X, metric=metric)
    dy = pdist(Y, metric=metric)

    if rank:
        r, _ = spearmanr(dx, dy)
        return {"spearman_dist": float(r)}
    else:
        r, _ = pearsonr(dx, dy)
        return {"pearson_dist": float(r)}


def affine_fit_scores(X, Y):
    """
    Least-squares affine fit.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same number of rows")
    N = X.shape[0]

    Xa = np.hstack([X, np.ones((N, 1), dtype=X.dtype)])  # add column of ones for bias
    # Solve best fit for Xa @ B = Y where B contains W^T and b^T (added ones above for b^T)
    B, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    Yhat = Xa @ B
    W = B[:-1, :].T
    b = B[-1, :]
    
    # calculate errors
    resid = Y - Yhat   # real - prediction
    sse = float(np.sum(resid**2))
    # total sum of squares around mean (per standard R^2)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    sst = float(np.sum(Yc**2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0

    # frobenius norms for error matrix
    rel_fro_error = float(np.linalg.norm(resid, "fro") / (np.linalg.norm(Y, "fro") + 1e-12))

    return float(r2), rel_fro_error

@dataclass
class LinearityReport:
    N: int
    n: int
    m: int
    knn_scores: Dict[int, float]
    knn_avg: float
    knn_scores_percentage: Dict[str, float]
    knn_avg_percentage: float
    pearson_dist: float
    spearman_dist: float
    r2_affine: float
    rel_fro_error_affine: float


def evaluate_pairset(X, Y, ks =(5, 10, 20, 50), ks_percentage=(0.01, 0.02, 0.05, 0.1, 0.2), metric="cosine", knn_mode = "jaccard"):
    """
    Compute metrics for paired datasets. Returns an object with the results.
    """
    X = _as_2d_float(X)
    Y = _as_2d_float(Y)
    N, n = X.shape
    m = Y.shape[1]
    if Y.shape[0] != N:
        raise ValueError("X and Y must have same number of points")

    # calculate also wrt percentages, save all
    ks_wrt_percentages = []
    k2p = {}   # so that we can remap the names
    for perc in ks_percentage:
        num = int(np.ceil(N*perc))
        ks_wrt_percentages.append(num)
        k2p[num] = f"{int(100*perc)}%"

    knn_scores = knn_overlap_scores_multi_k(X, Y, ks=ks, metric=metric, mode=knn_mode)
    knn_scores_percentages = knn_overlap_scores_multi_k(X,Y, ks=ks_wrt_percentages, metric=metric, mode=knn_mode)

    knn_avg = float(np.mean(list(knn_scores.values()))) if knn_scores else float("nan")
    knn_avg_percentages = float(np.mean(list(knn_scores_percentages.values()))) if knn_scores_percentages else float("nan")
    # remap percentages

    knn_scores_percentages = {k2p[k]:v for k, v in knn_scores_percentages.items()}
    pear = distance_correlation_scores(X, Y, metric=metric, rank=False)["pearson_dist"]
    spear = distance_correlation_scores(X, Y, metric=metric, rank=True)["spearman_dist"]
    aff_r2, aff_rel_fro = affine_fit_scores(X, Y)


    return LinearityReport(
        N=N, n=n, m=m,
        knn_scores=knn_scores,
        knn_avg=knn_avg,
        knn_scores_percentage=knn_scores_percentages,
        knn_avg_percentage=knn_avg_percentages,
        pearson_dist=float(pear),
        spearman_dist=float(spear),
        r2_affine=float(aff_r2),
        rel_fro_error_affine=float(aff_rel_fro),
    )