"""Inference engine powered by ONNX Runtime and Scikit-Learn PCA."""

import io
import pickle
from typing import Optional
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.decomposition import PCA

from app.config import settings


class RecommenderEngine:

    def __init__(self) -> None:
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.pca: Optional[PCA] = None

    def load_resources(self) -> None:
        """Loads ONNX inference session and the pre-fitted PCA transformer."""
        # 1. Initialize ONNX Runtime Inference Session
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(settings.ONNX_MODEL_PATH),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        # Warm-up pass
        dummy_input = np.zeros(
            (1, settings.IMAGE_SIZE[0], settings.IMAGE_SIZE[1], 3),
            dtype=np.float32,
        )
        self.session.run(None, {self.input_name: dummy_input})

        # 2. Load pre-fitted PCA model
        if settings.PCA_MODEL_PATH.exists():
            with open(settings.PCA_MODEL_PATH, "rb") as f:
                self.pca = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"PCA artifact missing at {settings.PCA_MODEL_PATH}"
            )

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocesses image using standard ResNet50 caffe-style normalization (pure NumPy)."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize(settings.IMAGE_SIZE, Image.Resampling.BILINEAR)
            x = np.array(img, dtype=np.float32)

            # Expand batch dimension: (1, 224, 224, 3)
            x = np.expand_dims(x, axis=0)

            # Convert RGB to BGR
            x = x[..., ::-1]

            # Subtract ImageNet channel means (BGR order)
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68

            return x.astype(np.float32)
        except Exception as exc:
            raise ValueError(f"Invalid image format: {exc}") from exc

    def extract_and_project(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """Runs ONNX Inference (2048-d) -> PCA Projection (1024-d) -> L2 Normalization."""
        # Step A: ONNX forward pass (2048-d)
        outputs = self.session.run(None, {self.input_name: preprocessed_image})
        features_2048 = outputs[0].reshape(1, -1)

        # Step B: PCA compression (1024-d)
        features_1024 = self.pca.transform(features_2048)[0]

        # Step C: L2 normalization
        norm = np.linalg.norm(features_1024)
        if norm > 0:
            features_1024 = features_1024 / norm

        return features_1024.astype(np.float32)


recommender = RecommenderEngine()