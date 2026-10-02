import pickle
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

ARTIFACTS_DIR = Path("artifacts")
EMBEDDINGS_IN = ARTIFACTS_DIR / "embeddings.npy"
EMBEDDINGS_OUT = ARTIFACTS_DIR / "embeddings_1024.npy"
PCA_MODEL_OUT = ARTIFACTS_DIR / "pca_1024.pkl"

TARGET_DIM = 1024


def main():
    print(f"Loading raw embeddings from {EMBEDDINGS_IN}...")
    features_2048 = np.load(EMBEDDINGS_IN).astype(np.float32)
    print(f"Original shape: {features_2048.shape}")

    # 1. Fit PCA and project 2048 -> 1024
    print(f"Fitting PCA with {TARGET_DIM} components...")
    pca = PCA(n_components=TARGET_DIM, random_state=42)
    features_1024 = pca.fit_transform(features_2048)

    # 2. Re-normalize to unit length for cosine similarity
    print("Re-normalizing transformed vectors (L2 norm)...")
    norms = np.linalg.norm(features_1024, axis=1, keepdims=True)
    features_1024 = features_1024 / np.maximum(norms, 1e-12)

    # 3. Save compressed embeddings and PCA model
    np.save(EMBEDDINGS_OUT, features_1024.astype(np.float32))
    with open(PCA_MODEL_OUT, "wb") as f:
        pickle.dump(pca, f)

    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Done! Explained variance retained: {explained_var:.2f}%")
    print(f"Saved compressed embeddings: {EMBEDDINGS_OUT}")
    print(f"Saved PCA transformer: {PCA_MODEL_OUT}")


if __name__ == "__main__":
    main()