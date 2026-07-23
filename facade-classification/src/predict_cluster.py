import argparse
import os
import numpy as np
import csv

from symmetry_features import extract_features

def load_model(model_path: str):
    d = np.load(model_path, allow_pickle=True)
    model = {
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
    return model

def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / (sd + 1e-8)

def pca_apply_cov(Xs: np.ndarray, pca_mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    # PCA projection: (Xs - mean) @ components^T
    Xc = Xs - pca_mean
    return (Xc @ components.T).astype(np.float32)

def assign_cluster(Xc: np.ndarray, centers: np.ndarray):
    dist2 = np.sum((Xc[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # n x k
    labels = np.argmin(dist2, axis=1).astype(np.int32)

    # confidence: 1 - d1/(d1+d2)
    part = np.partition(dist2, 1, axis=1)
    d1 = dist2[np.arange(Xc.shape[0]), labels]
    d2 = part[:, 1]
    conf = 1.0 - (d1 / (d1 + d2 + 1e-8))
    return labels, conf.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="cluster_model.npz")
    ap.add_argument("--image_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="new_clusters.csv")
    args = ap.parse_args()

    model = load_model(args.model)

    # Collect images
    exts = (".jpg", ".jpeg", ".png")
    paths = [os.path.join(args.image_dir, f) for f in sorted(os.listdir(args.image_dir)) if f.lower().endswith(exts)]
    if not paths:
        raise RuntimeError("No images found in image_dir.")

    # Extract features
    X = []
    imgs = []
    for p in paths:
        feats = extract_features(p)
        x = np.array([feats[c] for c in model["feature_cols"]], dtype=np.float32)
        X.append(x)
        imgs.append(os.path.basename(p))

    X = np.stack(X, axis=0)  # n x d

    # Apply standardization fitted on training set
    Xs = standardize_apply(X, model["mu"], model["sd"])

    # Apply clustering space transform
    if model["space"] == "pca2":
        X_cluster = pca_apply_cov(Xs, model["pca_mean"], model["pca_components"])
    else:
        X_cluster = Xs

    labels, conf = assign_cluster(X_cluster, model["centers"])

    # Write output
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "cluster", "cluster_name", "confidence", "reason"])
        w.writeheader()
        for i, img in enumerate(imgs):
            j = int(labels[i])
            w.writerow({
                "image": img,
                "cluster": j,
                "cluster_name": model["cluster_names"][j],
                "confidence": f"{float(conf[i]):.3f}",
                "reason": model["cluster_reasons"][j],
            })

    print(f"Wrote {args.out_csv}")
    print("Example predictions:")
    for i in range(min(5, len(imgs))):
        j = int(labels[i])
        print(f"  {imgs[i]} -> {model['cluster_names'][j]} (conf {float(conf[i]):.2f})")

if __name__ == "__main__":
    main()
