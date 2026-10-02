"""Script to upload local dataset images to Supabase Storage in parallel."""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client, Client
from tqdm import tqdm
from app.config import settings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# --- 1. Supabase Credentials ---
SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_KEY = settings.SUPABASE_KEY
BUCKET_NAME = settings.BUCKET_NAME

# --- 2. Local Dataset Path ---
# Point to your local extracted images directory
LOCAL_IMAGES_DIR = ROOT_DIR / "myntradataset/images"

# Number of parallel network worker threads
MAX_WORKERS = 20

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def upload_single_file(file_path: Path):
    """Uploads one image file to Supabase Storage bucket."""
    filename = file_path.name
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Upload binary file directly
        supabase.storage.from_(BUCKET_NAME).upload(
            path=filename,
            file=file_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )
        return True, filename
    except Exception as e:
        # Ignore already existing files if resuming
        if "Duplicate" in str(e) or "already exists" in str(e):
            return True, filename
        return False, f"{filename}: {str(e)}"


def main():
    if not LOCAL_IMAGES_DIR.exists():
        print(f"Error: Directory {LOCAL_IMAGES_DIR} does not exist.")
        return

    # Gather all image files
    all_files = [
        f for f in LOCAL_IMAGES_DIR.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    print(f"Found {len(all_files)} images to upload.")

    success_count = 0
    failure_count = 0

    # Run multi-threaded parallel uploads
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(upload_single_file, f): f for f in all_files}

        with tqdm(total=len(all_files), desc="Uploading to Supabase") as pbar:
            for future in as_completed(futures):
                success, msg = future.result()
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                pbar.update(1)

    print(f"\nUpload finished! Success: {success_count}, Failed: {failure_count}")


if __name__ == "__main__":
    main()