from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendedItem(BaseModel):
    """Schema representing an individual recommended item with product details."""

    rank: int = Field(..., description="Rank position (1 to K)")
    product_id: str = Field(..., description="Product identifier")
    filename: str = Field(..., description="Image filename")

    # Public CDN Image URL
    image_url: str = Field(..., description="Direct CDN URL to the product image asset")

    similarity_score: float = Field(..., description="Cosine similarity score")
    product_name: Optional[str] = Field(default="Unknown", description="Display title of the product.")
    gender: Optional[str] = Field(default=None, description="Target gender (Men, Women, etc.).")
    master_category: Optional[str] = Field(default=None, description="High-level category (Apparel, Accessories, etc.).")
    sub_category: Optional[str] = Field(default=None, description="Sub-category (Topwear, Bottomwear, etc.).")
    article_type: Optional[str] = Field(default=None, description="Detailed type (Shirts, Tshirts, Jeans, etc.).")
    base_colour: Optional[str] = Field(default=None, description="Primary color.")
    season: Optional[str] = Field(default=None, description="Season (Fall, Summer, etc.).")
    year: Optional[int] = Field(default=None, description="Release year.")
    usage: Optional[str] = Field(default=None, description="Usage context (Casual, Formal, etc.).")


class RecommendationResponse(BaseModel):
    """Root response schema for visual search requests."""

    total_results: int = Field(..., description="Number of recommendations returned.")
    query_image_name: str = Field(..., description="Original filename of the uploaded query image.")
    results: List[RecommendedItem] = Field(
        default_factory=list,
        description="List of top similar product matches.",
    )


class HealthCheckResponse(BaseModel):
    """Schema for server readiness and liveness probes."""

    status: str
    total_indexed_items: int
    embedding_dimension: int
    model_loaded: bool

class ProductUpdatePayload(BaseModel):
    """Payload for updating product metadata via PATCH."""

    product_name: Optional[str] = Field(None, description="Updated display title.")
    gender: Optional[str] = Field(None, description="Target gender.")
    master_category: Optional[str] = Field(None, description="Master category.")
    sub_category: Optional[str] = Field(None, description="Sub-category.")
    article_type: Optional[str] = Field(None, description="Article type.")
    base_colour: Optional[str] = Field(None, description="Primary color.")
    season: Optional[str] = Field(None, description="Season.")
    year: Optional[int] = Field(None, description="Release year.")
    usage: Optional[str] = Field(None, description="Usage classification.")


class AdminActionResponse(BaseModel):
    """Generic status response for admin mutations."""

    success: bool
    message: str
    product_id: str