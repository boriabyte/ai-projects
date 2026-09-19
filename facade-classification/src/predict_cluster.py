import argparse
import os
import numpy as np
import csv

from symmetry_features import extract_features


def load_model(model_path: str):
    d = np.load(model_path, allow_pickle=True)

    return {
        "k": int(d["k"][0]),
        "feature_cols": list(d["feature_cols"]),
        "mu": d["mu"],
        "sd": d["sd"],
        "pca_mean": d["pca_mean"],
        "pca_components": d["pca_components"],
        "space": str(d["space"][0]),
        "centers": d["centers"],
        "centers_feat": d["centers_feat"],
        "cluster_names": list(d["cluster_names"]),
        "cluster_reasons": list(d["cluster_reasons"]),
    }


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / (sd + 1e-8)


def pca_apply_cov(
    Xs: np.ndarray,
    pca_mean: np.ndarray,
    components: np.ndarray
) -> np.ndarray:
    Xc = Xs - pca_mean
    return (Xc @ components.T).astype(np.float32)


def assign_cluster(Xc: np.ndarray, centers: np.ndarray):
    dist2 = np.sum(
        (Xc[:, None, :] - centers[None, :, :]) ** 2,
        axis=2
    )

    labels = np.argmin(dist2, axis=1).astype(np.int32)

    if centers.shape[0] < 2:
        raise ValueError("At least two clusters are required.")

    part = np.partition(dist2, kth=1, axis=1)

    d1 = dist2[np.arange(Xc.shape[0]), labels]
    d2 = part[:, 1]

    separation = d2 / (d1 + d2 + 1e-8)

    return (
        labels,
        separation.astype(np.float32),
        d1.astype(np.float32),
        d2.astype(np.float32),
        dist2.astype(np.float32)
    )


def get_assignment_status(separation: float) -> str:
    if separation < 0.55:
        return "ambiguous"

    if separation < 0.65:
        return "weak"

    return "clear"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        type=str,
        default="cluster_model.npz"
    )

    ap.add_argument(
        "--image_dir",
        type=str,
        required=True
    )

    ap.add_argument(
        "--out_csv",
        type=str,
        default="new_clusters.csv"
    )

    ap.add_argument(
        "--verbose",
        action="store_true"
    )

    args = ap.parse_args()

    model = load_model(args.model)

    exts = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    paths = [
        os.path.join(args.image_dir, f)
        for f in sorted(os.listdir(args.image_dir))
        if f.lower().endswith(exts)
    ]

    if not paths:
        raise RuntimeError("No images found in image_dir.")

    X = []
    imgs = []

    for p in paths:
        feats = extract_features(p)

        missing_features = [
            c
            for c in model["feature_cols"]
            if c not in feats
        ]

        if missing_features:
            raise KeyError(
                f"Image '{p}' is missing required features: {missing_features}"
            )

        x = np.array(
            [feats[c] for c in model["feature_cols"]],
            dtype=np.float32
        )

        if not np.all(np.isfinite(x)):
            raise ValueError(
                f"Non-finite feature value detected for image '{p}'."
            )

        X.append(x)
        imgs.append(os.path.basename(p))

    X = np.stack(X, axis=0)

    Xs = standardize_apply(
        X,
        model["mu"],
        model["sd"]
    )

    if model["space"] == "pca2":
        X_cluster = pca_apply_cov(
            Xs,
            model["pca_mean"],
            model["pca_components"]
        )
    elif model["space"] in ("standardized", "zscore"):
        X_cluster = Xs
    else:
        raise ValueError(
            f"Unknown clustering space: {model['space']}"
        )

    labels, separation, d1, d2, all_distances = assign_cluster(
        X_cluster,
        model["centers"]
    )

    fieldnames = [
        "image",
        "cluster",
        "cluster_name",
        "separation_score",
        "assignment_status",
        "distance_nearest",
        "distance_second",
        "cluster_reason"
    ]

    with open(
        args.out_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        w.writeheader()

        for i, img in enumerate(imgs):
            j = int(labels[i])
            score = float(separation[i])
            status = get_assignment_status(score)

            w.writerow({
                "image": img,
                "cluster": j,
                "cluster_name": model["cluster_names"][j],
                "separation_score": f"{score:.3f}",
                "assignment_status": status,
                "distance_nearest": f"{float(d1[i]):.6f}",
                "distance_second": f"{float(d2[i]):.6f}",
                "cluster_reason": model["cluster_reasons"][j],
            })

    print(f"Wrote {args.out_csv}")
    print()
    print("Predictions:")

    for i, img in enumerate(imgs):
        j = int(labels[i])
        score = float(separation[i])
        status = get_assignment_status(score)

        print(
            f"{img} -> "
            f"{model['cluster_names'][j]} "
            f"[{status}] "
            f"(separation={score:.3f}, "
            f"d1={float(d1[i]):.4f}, "
            f"d2={float(d2[i]):.4f})"
        )

        if args.verbose:
            print("  Raw features:")

            for name, value in zip(
                model["feature_cols"],
                X[i]
            ):
                print(
                    f"    {name}: {float(value):.6f}"
                )

            print("  Standardized features:")

            for name, value in zip(
                model["feature_cols"],
                Xs[i]
            ):
                print(
                    f"    {name}: {float(value):.6f}"
                )

            print("  Distances to all clusters:")

            for cluster_idx, distance in enumerate(
                all_distances[i]
            ):
                cluster_name = model["cluster_names"][cluster_idx]

                print(
                    f"    Cluster {cluster_idx} "
                    f"({cluster_name}): "
                    f"{float(distance):.6f}"
                )

            print("  Cluster-space vector:")
            print("   ", X_cluster[i])
            print()


if __name__ == "__main__":
    main()