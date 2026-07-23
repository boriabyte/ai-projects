import csv
import numpy as np


# Must match FEATURE_COLS used in training
FEATURE_COLS = ["S_ref", "S_per_x", "S_per_y", "edge_density", "anisotropy"]


def read_features_csv(path: str, feature_cols=FEATURE_COLS):
    images = []
    X = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            images.append(row["image"])
            X.append([float(row[c]) for c in feature_cols])
    return images, np.asarray(X, dtype=np.float32)


def read_assignments_csv(path: str):
    # Works for clusters.csv / new_clusters.csv
    images = []
    labels = []
    conf = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            images.append(row["image"])
            labels.append(int(row["cluster"]))
            conf.append(float(row["confidence"]))
    return images, np.asarray(labels, dtype=np.int32), np.asarray(conf, dtype=np.float32)


def load_model_npz(model_path: str):
    d = np.load(model_path, allow_pickle=True)
    model = {
        "k": int(d["k"][0]),
        "feature_cols": list(d["feature_cols"]),
        "mu": d["mu"],              # (1,d)
        "sd": d["sd"],              # (1,d)
        "space": str(d["space"][0]),  # "zscore" or "pca2"
        "pca_mean": d.get("pca_mean", None),
        "pca_components": d.get("pca_components", None),  # (2,d)
        "pca_evr": d.get("pca_evr", None),               # (2,)
    }
    return model


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / (sd + 1e-8)


def pca_apply(Xs: np.ndarray, pca_mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    # (Xs - mean) @ components^T
    Xc = Xs - pca_mean
    return (Xc @ components.T).astype(np.float32)


def to_cluster_space(X: np.ndarray, model: dict) -> np.ndarray:
    # Ensure feature order matches model
    # (If your features.csv is in the same FEATURE_COLS order, this is already aligned.)
    Xs = standardize_apply(X, model["mu"], model["sd"])
    if model["space"] == "pca2":
        return pca_apply(Xs, model["pca_mean"], model["pca_components"])
    return Xs


def pairwise_dist_row(X: np.ndarray, i: int) -> np.ndarray:
    # distances from X[i] to all points (O(n*d), memory O(n))
    diff = X - X[i]
    return np.sqrt(np.sum(diff * diff, axis=1))


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean silhouette over all samples.
    Implementation is O(n^2) time; for typical project datasets (hundreds / low thousands) it's fine.
    """
    n = X.shape[0]
    uniq = np.unique(labels)
    # Precompute indices per cluster for speed
    idx = {c: np.where(labels == c)[0] for c in uniq}

    s_all = np.zeros((n,), dtype=np.float32)

    for i in range(n):
        ci = labels[i]
        d = pairwise_dist_row(X, i)

        # a(i): mean distance to same cluster (excluding itself)
        same = idx[ci]
        if same.size <= 1:
            a = 0.0
        else:
            a = float(np.sum(d[same]) - 0.0) / float(same.size - 1)  # d[i]=0, subtracting not needed

        # b(i): minimum mean distance to other clusters
        b = np.inf
        for cj in uniq:
            if cj == ci:
                continue
            other = idx[cj]
            if other.size > 0:
                b = min(b, float(np.mean(d[other])))

        denom = max(a, b) + 1e-8
        s_all[i] = (b - a) / denom

    return float(np.mean(s_all))


def davies_bouldin_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    DB index: lower is better.
    """
    uniq = np.unique(labels)
    k = uniq.size

    centroids = np.zeros((k, X.shape[1]), dtype=np.float32)
    scat = np.zeros((k,), dtype=np.float32)

    for t, c in enumerate(uniq):
        Xi = X[labels == c]
        centroids[t] = np.mean(Xi, axis=0)
        # scatter = mean distance to centroid
        dif = Xi - centroids[t]
        scat[t] = float(np.mean(np.sqrt(np.sum(dif * dif, axis=1)) + 1e-8))

    # pairwise centroid distances
    D = np.sqrt(np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2)) + 1e-8

    R = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            if i == j:
                R[i, j] = 0.0
            else:
                R[i, j] = (scat[i] + scat[j]) / D[i, j]

    # DB = mean over clusters of max_j R_ij
    return float(np.mean(np.max(R, axis=1)))


def confidence_summary(labels: np.ndarray, conf: np.ndarray):
    uniq = np.unique(labels)
    out = {
        "mean": float(np.mean(conf)),
        "median": float(np.median(conf)),
        "p10": float(np.percentile(conf, 10)),
        "p90": float(np.percentile(conf, 90)),
        "per_cluster_mean": {int(c): float(np.mean(conf[labels == c])) for c in uniq},
        "per_cluster_count": {int(c): int(np.sum(labels == c)) for c in uniq},
    }
    return out


def main(
    features_csv="features.csv",
    assignments_csv="clusters.csv",
    model_npz="cluster_model.npz",
):
    # Load data
    imgF, X = read_features_csv(features_csv, FEATURE_COLS)
    imgA, labels, conf = read_assignments_csv(assignments_csv)

    # Basic alignment check (optional but helpful)
    if imgF != imgA:
        print("Warning: image order mismatch between features and assignments CSVs.")
        # If needed, you can build a dict and reorder; for now we assume they match.

    # Map X into the *same clustering space used by training*
    model = load_model_npz(model_npz)
    Xc = to_cluster_space(X, model)

    # Metrics
    sil = silhouette_score(Xc, labels)
    db = davies_bouldin_index(Xc, labels)

    print("=== Unsupervised clustering metrics ===")
    print(f"Silhouette score:        {sil:.4f}  (higher is better)")
    print(f"Davies–Bouldin index:    {db:.4f}  (lower is better)")

    if model.get("pca_evr", None) is not None:
        evr = model["pca_evr"].astype(np.float32)
        print(f"PCA explained variance:  PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%, sum={(evr[0]+evr[1])*100:.1f}%")

    cs = confidence_summary(labels, conf)
    print("\n=== Assignment confidence summary ===")
    print(f"Mean={cs['mean']:.3f} | Median={cs['median']:.3f} | P10={cs['p10']:.3f} | P90={cs['p90']:.3f}")
    print("Per-cluster mean confidence / count:")
    for c in sorted(cs["per_cluster_mean"].keys()):
        print(f"  cluster {c}: mean_conf={cs['per_cluster_mean'][c]:.3f}, n={cs['per_cluster_count'][c]}")


if __name__ == "__main__":
    main()
