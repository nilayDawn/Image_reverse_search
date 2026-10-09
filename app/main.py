from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from supabase import Client, create_client

from app.config import settings
from app.model import recommender
from app.schemas import HealthCheckResponse, RecommendationResponse, RecommendedItem

from app.admin import admin_router

# Initialize Supabase Client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    recommender.load_resources()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan
)

cors_origins = settings.cors_origins_list

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if "*" not in cors_origins else ["*"],
    allow_credentials=True if "*" not in cors_origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_router)

def _process_image_to_vector(image_bytes: bytes) -> list[float]:
    preprocessed = recommender.preprocess_image(image_bytes)
    vector_1024 = recommender.extract_and_project(preprocessed)
    return vector_1024.tolist()


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    tags=["Recommendation"],
    summary="Recommend products by visual similarity with automatic filter-based fallback",
)
async def recommend(
    file: Annotated[UploadFile, File(...)],
    top_k: Annotated[int, Query(ge=1, le=settings.MAX_TOP_K)] = (
        settings.DEFAULT_TOP_K
    ),
    match_threshold: Annotated[
        float,
        Query(
            ge=0.0,
            le=1.0,
            description="Minimum cosine similarity required to consider an image match.",
        ),
    ] = 0.50,
    # Optional metadata filters
    gender: Annotated[
        Optional[str],
        Query(description="Filter by gender (e.g. 'Men', 'Women')"),
    ] = None,
    master_category: Annotated[
        Optional[str],
        Query(description="Filter by master category (e.g. 'Apparel')"),
    ] = None,
    sub_category: Annotated[
        Optional[str],
        Query(description="Filter by sub-category (e.g. 'Topwear')"),
    ] = None,
    article_type: Annotated[
        Optional[str],
        Query(description="Filter by article type (e.g. 'Shirts')"),
    ] = None,
    base_colour: Annotated[
        Optional[str],
        Query(description="Filter by color (e.g. 'Blue', 'Black')"),
    ] = None,
) -> RecommendationResponse:
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp",
    ]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file format",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Extract 1024-d unit vector via ONNX + PCA
    query_vector = await run_in_threadpool(
        _process_image_to_vector, image_bytes
    )

    # 2. Attempt visual search via match_products
    rpc_params = {
        "query_embedding": query_vector,
        "match_threshold": match_threshold,
        "match_count": top_k,
        "filter_gender": gender,
        "filter_master_category": master_category,
        "filter_sub_category": sub_category,
        "filter_article_type": article_type,
        "filter_base_colour": base_colour,
    }

    try:
        response = supabase.rpc("match_products", rpc_params).execute()
        rows = response.data or []
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database search failed: {str(exc)}",
        )

    search_mode = "visual_match"

    # 3. Fallback logic: If no visual matches found AND at least one filter was provided
    has_any_filter = any(
        [gender, master_category, sub_category, article_type, base_colour]
    )

    if not rows and has_any_filter:
        fallback_params = {
            "filter_gender": gender,
            "filter_master_category": master_category,
            "filter_sub_category": sub_category,
            "filter_article_type": article_type,
            "filter_base_colour": base_colour,
            "match_count": top_k,
        }
        try:
            fallback_res = supabase.rpc(
                "filter_products_only", fallback_params
            ).execute()
            rows = fallback_res.data or []
            if rows:
                search_mode = "filter_fallback"
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Fallback search failed: {str(exc)}",
            )

    # 4. Format recommendations payload
    recommendations = [
        RecommendedItem(
            rank=idx,
            product_id=row["id"],
            filename=f"{row['id']}.jpg",
            image_url=row["image_url"],
            similarity_score=round(row["similarity"], 4),
            product_name=row["product_name"],
            gender=row["gender"],
            master_category=row["master_category"],
            sub_category=row["sub_category"],
            article_type=row["article_type"],
            base_colour=row["base_colour"],
            season=row["season"],
            year=row["year"],
            usage=row["usage"],
        )
        for idx, row in enumerate(rows, start=1)
    ]

    return RecommendationResponse(
        total_results=len(recommendations),
        query_image_name=file.filename or "unknown.jpg",
        search_mode=search_mode,
        results=recommendations,
    )
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Monitoring"],
    summary="Health and readiness probe",
)
async def health_check() -> HealthCheckResponse:
    """Verifies that the API is alive and the FAISS index is loaded in memory."""
    is_ready = recommender.index is not None and recommender.index.ntotal > 0
    return HealthCheckResponse(
        status="ready" if is_ready else "not_ready",
        total_indexed_items=recommender.index.ntotal if is_ready else 0,
        embedding_dimension=recommender.embedding_dim if is_ready else 0,
        model_loaded=recommender.model is not None,
    )