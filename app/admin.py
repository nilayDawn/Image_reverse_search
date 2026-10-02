"""Admin endpoints for managing catalog products and embeddings."""

import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool
from supabase import Client, create_client

from app.config import settings
from app.model import recommender
from app.schemas import AdminActionResponse, ProductUpdatePayload
from app.security import verify_admin_key

admin_router = APIRouter(
    prefix="/admin",
    tags=["Catalog Administration"],
    dependencies=[Depends(verify_admin_key)],
)

supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


def _process_image(image_bytes: bytes) -> list[float]:
    """Runs preprocessing, ResNet50 extraction, PCA projection, and L2 normalization."""
    preprocessed = recommender.preprocess_image(image_bytes)
    return recommender.extract_and_project(preprocessed).tolist()


@admin_router.post(
    "/products",
    response_model=AdminActionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a new product",
)
async def create_product(
    file: Annotated[
        UploadFile, File(description="Product photo (.jpg, .jpeg, .png)")
    ],
    product_name: Annotated[str, Form()],
    gender: Annotated[Optional[str], Form()] = None,
    master_category: Annotated[Optional[str], Form()] = None,
    sub_category: Annotated[Optional[str], Form()] = None,
    article_type: Annotated[Optional[str], Form()] = None,
    base_colour: Annotated[Optional[str], Form()] = None,
    season: Annotated[Optional[str], Form()] = None,
    year: Annotated[Optional[int], Form()] = None,
    usage: Annotated[Optional[str], Form()] = None,
) -> AdminActionResponse:
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File must be a JPEG, PNG, or WebP image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file.")

    # 1. Compute 1024-d unit vector via model & PCA
    embedding = await run_in_threadpool(_process_image, image_bytes)

    # 2. Build unique ID and filename
    product_id = str(uuid.uuid4())[:8]
    ext = file.filename.split(".")[-1] if file.filename and "." in file.filename else "jpg"
    storage_path = f"{product_id}.{ext}"

    # 3. Store asset in Supabase Bucket
    try:
        supabase.storage.from_(settings.STORAGE_BUCKET).upload(
            path=storage_path,
            file=image_bytes,
            file_options={"content-type": file.content_type or "image/jpeg", "upsert": "true"},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Storage upload failed: {str(exc)}",
        )

    image_url = f"{settings.CDN_BASE_URL}/{storage_path}"

    # 4. Insert row into PostgreSQL pgvector table
    record = {
        "id": product_id,
        "product_name": product_name,
        "gender": gender,
        "master_category": master_category,
        "sub_category": sub_category,
        "article_type": article_type,
        "base_colour": base_colour,
        "season": season,
        "year": year,
        "usage": usage,
        "image_url": image_url,
        "embedding": embedding,
    }

    db_response = supabase.table("products").insert(record).execute()
    if not db_response.data:
        # Rollback storage asset if database write fails
        supabase.storage.from_(settings.STORAGE_BUCKET).remove([storage_path])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to insert product record into database.",
        )

    return AdminActionResponse(
        success=True,
        message="Product successfully uploaded, indexed, and available for visual search.",
        product_id=product_id,
    )


@admin_router.patch(
    "/products/{product_id}",
    response_model=AdminActionResponse,
    summary="Update product catalog metadata",
)
async def update_product(
    product_id: str,
    payload: ProductUpdatePayload,
) -> AdminActionResponse:
    # Filter out fields that were not provided in the request
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid fields provided for update.",
        )

    db_response = supabase.table("products").update(updates).eq("id", product_id).execute()
    if not db_response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID '{product_id}' not found.",
        )

    return AdminActionResponse(
        success=True,
        message="Product metadata updated.",
        product_id=product_id,
    )


@admin_router.delete(
    "/products/{product_id}",
    response_model=AdminActionResponse,
    summary="Delete a product and its storage assets",
)
async def delete_product(product_id: str) -> AdminActionResponse:
    # Fetch row to determine file name
    existing = supabase.table("products").select("image_url").eq("id", product_id).execute()
    if not existing.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID '{product_id}' not found.",
        )

    # 1. Delete database record
    supabase.table("products").delete().eq("id", product_id).execute()

    # 2. Extract filename from image_url and prune from storage
    image_url = existing.data[0].get("image_url", "")
    filename = image_url.split("/")[-1]
    if filename:
        try:
            supabase.storage.from_(settings.STORAGE_BUCKET).remove([filename])
        except Exception:
            pass  # Best effort storage cleanup

    return AdminActionResponse(
        success=True,
        message="Product and associated storage assets deleted.",
        product_id=product_id,
    )