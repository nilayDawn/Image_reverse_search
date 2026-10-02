"""Exports the ResNet50 feature extraction model to ONNX format compatible with Keras 3."""

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
import tf2onnx

ARTIFACTS_DIR = Path("artifacts")
ONNX_OUTPUT_PATH = ARTIFACTS_DIR / "resnet50_extractor.onnx"


def main():
    print("Building base Keras model (ResNet50 + GlobalMaxPooling2D)...")
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

    # Define concrete input signature: (batch_size=1, 224, 224, 3) float32
    input_signature = [
        tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32, name="input_tensor")
    ]

    # Wrap in tf.function to create a compiled TensorFlow graph (bypasses Keras 3 KerasTensor bug)
    @tf.function(input_signature=input_signature)
    def model_fn(input_tensor):
        return model(input_tensor, training=False)

    print("Converting compiled graph to ONNX format via from_function...")
    model_proto, _ = tf2onnx.convert.from_function(
        model_fn,
        input_signature=input_signature,
        opset=13,
        output_path=str(ONNX_OUTPUT_PATH),
    )

    print(f"SUCCESS: Model exported to {ONNX_OUTPUT_PATH}")


if __name__ == "__main__":
    main()