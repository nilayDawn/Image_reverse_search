from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendedItem(BaseModel):
    rank: int
    product_id: str
    filename: str
    image_url: str
    similarity_score: float
    product_name: str
    gender: Optional[str] = None
    master_category: Optional[str] = None
    sub_category: Optional[str] = None
    article_type: Optional[str] = None
    base_colour: Optional[str] = None
    season: Optional[str] = None
    year: Optional[int] = None
    usage: Optional[str] = None


class RecommendationResponse(BaseModel):
    total_results: int
    query_image_name: str
    search_mode: str = Field(
        ...,
        description="Indicates whether results are 'visual_match' or 'filter_fallback'.",
    )
    results: List[RecommendedItem]


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