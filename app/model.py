"""Feature extraction and vector search service module.

Handles:
1. Deep learning model initialization (ResNet50 + GlobalMaxPooling2D).
2. Pure in-memory image preprocessing (PIL -> Numpy -> Keras).
3. Pre-computed embedding indexing via FAISS IndexFlatIP (Cosine Similarity).
4. Top-K nearest neighbor querying.
"""

import io
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D

from app.config import settings
from app.schemas import RecommendedItem

from app.storage import storage_client


class RecommenderEngine:

    def __init__(self) -> None:
        self.model: tf.keras.Model | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.filenames: List[str] = []
        self.embedding_dim: int = 0
        self.metadata_lookup: Dict[str, dict] = {}

    def load_resources(self) -> None:
        """Loads model, FAISS index, filenames, and product metadata into RAM."""
        # 1. Rebuild feature extractor
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(settings.IMAGE_SIZE[0], settings.IMAGE_SIZE[1], 3),
        )
        base_model.trainable = False

        self.model = tf.keras.Sequential([
            base_model,
            GlobalMaxPooling2D(),
        ])

        # Warm-up pass
        dummy_input = np.zeros(
            (1, settings.IMAGE_SIZE[0], settings.IMAGE_SIZE[1], 3),
            dtype=np.float32,
        )
        _ = self.model(dummy_input, training=False)

        # 2. Load embeddings
        embeddings = np.load(settings.EMBEDDINGS_PATH).astype(np.float32)

        # 3. Load filenames
        with open(settings.FILENAMES_PATH, "rb") as f:
            raw_filenames = pickle.load(f)
        self.filenames = [os.path.basename(path) for path in raw_filenames]

        # 4. Build FAISS index
        self.embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        # 5. Load and index styles.csv metadata into an O(1) dictionary
        if settings.STYLES_PATH.exists():
            df = pd.read_csv(settings.STYLES_PATH, on_bad_lines="skip")
            # Convert 'id' column to string to match product_id
            df["id"] = df["id"].astype(str)
            # Store records as a dictionary keyed by product id
            self.metadata_lookup = df.set_index("id").to_dict(orient="index")
            print(f"Loaded metadata for {len(self.metadata_lookup)} products.")
        else:
            print(
                f"Warning: {settings.STYLES_PATH} not found. Running without metadata."
            )

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize(settings.IMAGE_SIZE, Image.Resampling.BILINEAR)
            img_array = np.array(img, dtype=np.float32)
            expanded_img = np.expand_dims(img_array, axis=0)
            return preprocess_input(expanded_img)
        except Exception as exc:
            raise ValueError(f"Invalid image format: {exc}") from exc

    def extract_features(self, preprocessed_image: np.ndarray) -> np.ndarray:
        features = (
            self.model(preprocessed_image, training=False).numpy().flatten()
        )
        norm = np.linalg.norm(features)
        if norm == 0:
            return features.astype(np.float32)
        return (features / norm).astype(np.float32)

    def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> List[RecommendedItem]:
        query_batch = np.expand_dims(query_vector, axis=0)
        distances, indices = self.index.search(query_batch, top_k)

        top_scores = distances[0]
        top_indices = indices[0]

        recommendations: List[RecommendedItem] = []

        for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
            if idx == -1:
                continue

            filename = self.filenames[idx]
            product_id = Path(filename).stem
            meta = self.metadata_lookup.get(product_id, {})

            # Resolve URL using the pluggable storage provider
            resolved_image_url = storage_client.get_image_url(filename)

            recommendations.append(
                RecommendedItem(
                    rank=rank,
                    product_id=product_id,
                    filename=filename,
                    image_url=resolved_image_url,  # <--- Cleanly injected
                    similarity_score=float(round(score, 4)),
                    product_name=meta.get("productDisplayName", "Unknown"),
                    gender=meta.get("gender"),
                    master_category=meta.get("masterCategory"),
                    sub_category=meta.get("subCategory"),
                    article_type=meta.get("articleType"),
                    base_colour=meta.get("baseColour"),
                    season=meta.get("season"),
                    year=int(meta["year"]) if pd.notna(meta.get("year")) else None,
                    usage=meta.get("usage"),
                )
            )

        return recommendations


recommender = RecommenderEngine()