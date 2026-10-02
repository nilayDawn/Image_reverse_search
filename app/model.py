import io
import pickle
from typing import Optional
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D

from app.config import settings


class RecommenderEngine:

    def __init__(self) -> None:
        self.model: Optional[tf.keras.Model] = None
        self.pca: Optional[PCA] = None

    def load_resources(self) -> None:
        """Loads ResNet50 and the pre-fitted PCA model into RAM."""
        # 1. Feature Extractor
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(settings.IMAGE_SIZE[0], settings.IMAGE_SIZE[1], 3),
        )
        base_model.trainable = False

        self.model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

        # Warm-up pass
        dummy_input = np.zeros(
            (1, settings.IMAGE_SIZE[0], settings.IMAGE_SIZE[1], 3),
            dtype=np.float32,
        )
        _ = self.model(dummy_input, training=False)

        # 2. Load pre-fitted PCA transformer
        if settings.PCA_MODEL_PATH.exists():
            with open(settings.PCA_MODEL_PATH, "rb") as f:
                self.pca = pickle.load(f)
            print(
                f"Loaded PCA model. Components: {self.pca.n_components_} dimensions."
            )
        else:
            raise FileNotFoundError(
                f"PCA artifact not found at {settings.PCA_MODEL_PATH}. Run scripts/train_pca.py first."
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

    def extract_and_project(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """Runs ResNet50 (2048-d) -> PCA (1024-d) -> L2 Normalization."""
        # Step A: 2048-d extraction
        features_2048 = (
            self.model(preprocessed_image, training=False).numpy().reshape(1, -1)
        )

        # Step B: Project to 1024-d using pre-fitted PCA
        features_1024 = self.pca.transform(features_2048)[0]

        # Step C: Re-normalize
        norm = np.linalg.norm(features_1024)
        if norm > 0:
            features_1024 = features_1024 / norm

        return features_1024.astype(np.float32)


recommender = RecommenderEngine()