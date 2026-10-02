"""Storage service abstraction (Strategy Pattern).

Allows swapping storage backends (Cloudflare R2, AWS S3, Supabase, Local)
by modifying .env without rewriting business logic.
"""

from abc import ABC, abstractmethod
from app.config import settings


class BaseStorageProvider(ABC):
    """Abstract base class for image asset URL resolvers."""

    @abstractmethod
    def get_image_url(self, filename: str) -> str:
        """Resolves a product filename to its publicly accessible URL."""
        pass


class CDNStorageProvider(BaseStorageProvider):
    """Resolves URLs for Cloudflare R2, S3, or CloudFront CDN."""

    def __init__(self, base_url: str):
        # Strip trailing slashes for clean URL concatenation
        self.base_url = base_url.rstrip("/")

    def get_image_url(self, filename: str) -> str:
        return f"{self.base_url}/{filename}"


class LocalStorageProvider(BaseStorageProvider):
    """Resolves URLs served directly by FastAPI /static mount during local development."""

    def __init__(self, host_url: str = "http://localhost:8000"):
        self.host_url = host_url.rstrip("/")

    def get_image_url(self, filename: str) -> str:
        return f"{self.host_url}/static/images/{filename}"


def get_storage_provider() -> BaseStorageProvider:
    """Factory function returning the active storage driver configured in settings."""
    backend = settings.STORAGE_BACKEND.lower()

    if backend == "cdn":
        return CDNStorageProvider(base_url=settings.CDN_BASE_URL)
    elif backend == "local":
        return LocalStorageProvider()
    else:
        # Fallback to CDN
        return CDNStorageProvider(base_url=settings.CDN_BASE_URL)


# Singleton instance used by recommendation service
storage_client = get_storage_provider()