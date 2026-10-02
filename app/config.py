"""Application configuration module.

Defines environment variables, file paths to artifacts, and model
hyperparameters.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    EMBEDDINGS_PATH: Path = ARTIFACTS_DIR / "embeddings.npy"
    FILENAMES_PATH: Path = ARTIFACTS_DIR / "filenames.pkl"
    STYLES_PATH: Path = ARTIFACTS_DIR / "styles.csv"

    IMAGE_SIZE: tuple[int, int] = (224, 224)
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 50

    PROJECT_NAME: str = "Fashion Similarity Recommender API"
    VERSION: str = "1.0.0"

    # --- Storage & CDN Configuration ---
    # Switch providers simply by changing STORAGE_BACKEND in .env
    # Options: "cdn" (Cloudflare/S3), "local", "mock"
    STORAGE_BACKEND: str = "cdn"

    # The public URL prefix provided by Cloudflare R2 / Custom domain
    # Example: "https://pub-xxxxxx.r2.dev" or "https://cdn.yourdomain.com"
    CDN_BASE_URL: str 

    SUPABASE_URL:str
    SUPABASE_KEY:str
    BUCKET_NAME:str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()