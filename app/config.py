from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    EMBEDDINGS_PATH: Path = ARTIFACTS_DIR / "embeddings_1024.npy"
    FILENAMES_PATH: Path = ARTIFACTS_DIR / "filenames.pkl"
    STYLES_PATH: Path = ARTIFACTS_DIR / "styles.csv"
    PCA_MODEL_PATH: Path = ARTIFACTS_DIR / "pca_1024.pkl"
    ONNX_MODEL_PATH: Path = ARTIFACTS_DIR / "resnet50_extractor.onnx"

    IMAGE_SIZE: tuple[int, int] = (224, 224)
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 50

    PROJECT_NAME: str = "Fashion Similarity Recommender API"
    VERSION: str = "1.0.0"

    STORAGE_BACKEND: str = "cdn"
    CDN_BASE_URL: str
    BUCKET_NAME: str 

    # Supabase Credentials
    SUPABASE_URL: str
    SUPABASE_KEY: str

    # Security
    ADMIN_API_KEY: str 

    DB_URI: str

    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parent.parent / ".env", env_file_encoding="utf-8")
    


settings = Settings() 