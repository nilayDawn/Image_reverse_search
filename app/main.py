"""FastAPI application entry point.

Exposes:
- GET  /health     : Liveness and readiness inspection.
- POST /recommend  : Visual similarity search from uploaded image.
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.model import recommender
from app.schemas import HealthCheckResponse, RecommendationResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager handling startup pre-loading and graceful teardown."""
    print("Initializing Recommender Engine and loading artifacts...")
    try:
        # Load model, embeddings, and FAISS index into RAM once
        recommender.load_resources()
        print(f"Indexed {len(recommender.filenames)} items (Dim: {recommender.embedding_dim}).")
    except Exception as exc:
        print(f"Failed to initialize recommender engine: {exc}")
        raise exc

    yield  # Application serves requests here

    # Clean up / shutdown logic if needed
    print("Shutting down service...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable Cross-Origin Resource Sharing (CORS) for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific domains in strict production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def _process_and_search(image_bytes: bytes, top_k: int):
    """Synchronous CPU pipeline run inside worker thread pool."""
    # 1. Decode & Preprocess image in-memory
    preprocessed_img = recommender.preprocess_image(image_bytes)

    # 2. Extract normalized vector
    query_vector = recommender.extract_features(preprocessed_img)

    # 3. Query FAISS index
    return recommender.search(query_vector=query_vector, top_k=top_k)


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    tags=["Recommendation"],
    summary="Find visually similar fashion products",
)
async def recommend(
    file: Annotated[UploadFile, File(description="Query image file (.jpg, .jpeg, .png)")],
    top_k: Annotated[
        int,
        Query(
            description="Number of similar items to return",
            ge=1,
            le=settings.MAX_TOP_K,
        ),
    ] = settings.DEFAULT_TOP_K,
) -> RecommendationResponse:
    """Accepts an uploaded image and returns the Top-K nearest visual matches."""
    # Validate content type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a JPEG or PNG image.",
        )

    try:
        # Read raw image bytes directly from memory stream
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        # Offload CPU-heavy inference and FAISS search to threadpool
        results = await run_in_threadpool(_process_and_search, image_bytes, top_k)

        return RecommendationResponse(
            total_results=len(results),
            query_image_name=file.filename or "unknown.jpg",
            results=results,
        )

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )