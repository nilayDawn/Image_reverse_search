import psycopg2
from app.config import settings

# Parse connection string or use direct Postgres URI from Supabase
# Settings -> Database -> Connection string -> URI
DB_URI = settings.DB_URI

def main():
    print("Connecting to remote Supabase database...")
    conn = psycopg2.connect(DB_URI)
    conn.autocommit = True
    cursor = conn.cursor()

    # 1. Clean up any invalid/partial index left behind by previous crashes
    print("Dropping any residual partial index...")
    cursor.execute("DROP INDEX IF EXISTS products_embedding_hnsw_idx;")

    # 2. Configure server session parameters:
    # Disable timeout completely for this script
    cursor.execute("SET statement_timeout = 0;")
    
    # 64MB fits comfortably inside Supabase's /dev/shm shared memory segment
    cursor.execute("SET maintenance_work_mem = '64MB';")
    
    # CRITICAL: 0 disables multi-process workers, preventing /dev/shm exhaustion
    cursor.execute("SET max_parallel_maintenance_workers = 0;")

    # 3. Build the HNSW index
    print("Building HNSW index sequentially (this will take 2-4 minutes)...")
    cursor.execute("""
        CREATE INDEX products_embedding_hnsw_idx 
        ON products 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 32);
    """)

    print("SUCCESS: HNSW index created successfully!")

    # Verify presence
    cursor.execute("""
        SELECT indexname, indexdef 
        FROM pg_indexes 
        WHERE tablename = 'products' AND indexname = 'products_embedding_hnsw_idx';
    """)
    result = cursor.fetchone()
    print("Registered Index Details:", result)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()