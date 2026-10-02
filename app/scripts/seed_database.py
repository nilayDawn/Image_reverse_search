import math
import pickle
from pathlib import Path
import time
import numpy as np
import pandas as pd
from supabase import Client, create_client
from tqdm import tqdm

from app.config import settings

# 40 rows per batch avoids hitting HTTP / statement timeout limits
BATCH_SIZE = 40
MAX_RETRIES = 3

supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


def sanitize_value(val):
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if pd.isna(val):
        return None
    return val


def insert_batch_with_retry(batch):
    """Retries a failed batch up to MAX_RETRIES times with exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            supabase.table("products").upsert(batch).execute()
            return
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise exc
            time.sleep(attempt * 2)


def main():
    print("Loading artifacts...")
    embeddings = np.load(settings.EMBEDDINGS_PATH)

    with open(settings.FILENAMES_PATH, "rb") as f:
        filenames = pickle.load(f)

    df = pd.read_csv(settings.STYLES_PATH, on_bad_lines="skip")
    df = df.where(pd.notnull(df), None)
    df["id"] = df["id"].astype(str)
    meta_lookup = df.set_index("id").to_dict(orient="index")

    rows = []
    print("Preparing rows for database...")
    for idx, path in enumerate(filenames):
        filename = Path(path).name
        prod_id = Path(path).stem

        meta = meta_lookup.get(prod_id, {})
        raw_year = sanitize_value(meta.get("year"))
        year_val = int(raw_year) if raw_year is not None else None

        row = {
            "id": prod_id,
            "product_name": sanitize_value(meta.get("productDisplayName")) or "Unknown",
            "gender": sanitize_value(meta.get("gender")),
            "master_category": sanitize_value(meta.get("masterCategory")),
            "sub_category": sanitize_value(meta.get("subCategory")),
            "article_type": sanitize_value(meta.get("articleType")),
            "base_colour": sanitize_value(meta.get("baseColour")),
            "season": sanitize_value(meta.get("season")),
            "year": year_val,
            "usage": sanitize_value(meta.get("usage")),
            "image_url": f"{settings.CDN_BASE_URL}/{filename}",
            "embedding": [float(x) for x in embeddings[idx]],
        }
        rows.append(row)

    print(f"Total rows to insert: {len(rows)}")

    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Seeding to Supabase"):
        batch = rows[i : i + BATCH_SIZE]
        insert_batch_with_retry(batch)

    print("All rows successfully seeded into Supabase!")


if __name__ == "__main__":
    main()