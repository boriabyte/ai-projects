import argparse
import numpy as np
import csv

from cluster_unsupervised import (
    FEATURE_COLS,
    read_features_csv,
    standardize_fit,
    pca_fit_transform_cov,
    kmeans,
    assign_user_label,
    cluster_reason,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", type=str, default="features.csv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_pca", action="store_true")
    ap.add_argument("--model_out", type=str, default="cluster_model.npz")
    args = ap.parse_args()

    images, X = read_features_csv(args.features_csv, FEATURE_COLS)

    # Fit standardization on training set
    Xs, mu, sd = standardize_fit(X)

    # Fit PCA (textbook) on standardized training set (used for plotting or clustering if requested)
    Z2, comps2, evr2, mean_pca = pca_fit_transform_cov(Xs, n_components=2, assume_centered=False)

    # Choose clustering space
    if args.use_pca:
        X_cluster = Z2
        space = "pca2"
    else:
        X_cluster = Xs
        space = "zscore"

    # Fit k-means on chosen space
    centers, labels = kmeans(X_cluster, k=args.k, seed=args.seed, iters=300)

    # cluster means in standardized ORIGINAL feature space
    centers_feat = np.zeros((args.k, len(FEATURE_COLS)), dtype=np.float32)
    for j in range(args.k):
        m = labels == j
        if np.any(m):
            centers_feat[j] = np.mean(Xs[m], axis=0)

    cluster_names = [assign_user_label(centers_feat[j], FEATURE_COLS) for j in range(args.k)]
    cluster_reasons = [cluster_reason(centers_feat[j], FEATURE_COLS, topn=2) for j in range(args.k)]

    # Save model
    np.savez(
        args.model_out,
        k=np.array([args.k], dtype=np.int32),
        feature_cols=np.array(FEATURE_COLS, dtype=object),

        # standardization
        mu=mu,
        sd=sd,

        # PCA (2D) for visualization / optional clustering
        pca_mean=mean_pca,      # mean used inside PCA on Xs
        pca_components=comps2,  # (2, d) components
        pca_evr=evr2,

        # clustering
        space=np.array([space], dtype=object),
        centers=centers,            # centers in chosen clustering space
        centers_feat=centers_feat,  # centers in standardized feature space for interpretation

        # names
        cluster_names=np.array(cluster_names, dtype=object),
        cluster_reasons=np.array(cluster_reasons, dtype=object),
        seed=np.array([args.seed], dtype=np.int32),
    )

    print(f"Saved model to {args.model_out}")
    print(f"Clustering space: {space}")
    print(f"Cluster names: {cluster_names}")

if __name__ == "__main__":
    main()
