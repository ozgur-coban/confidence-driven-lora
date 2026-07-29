import duckdb

remote_path = 'hf://datasets/allenai/WildChat-4.8M/**/*.parquet'
output_file = './data/wildchat/filtered_math_conversations.jsonl.gz'

con = duckdb.connect()

# --- THE FIXES ---
# 1. Limit the threads to 4 (prevents hammering HF with too many parallel requests)
con.execute("SET threads = 4;")

# 2. Increase the connection timeout (60 seconds instead of default 30)
con.execute("SET http_timeout = 60000;")

# 3. Tell DuckDB to retry up to 10 times if a connection drops
con.execute("SET http_retries = 10;")

# 4. Add a progressive backoff multiplier to retries
con.execute("SET http_retry_backoff = 2.0;")

# 5. Enable the progress bar
con.execute("SET enable_progress_bar = true;")

print("Searching for English conversations containing 'import math' (resilient mode)...")

query = f"""
    COPY (
        SELECT * 
        FROM '{remote_path}' 
        WHERE CAST(conversation AS VARCHAR) LIKE '%import math%'
          AND language = 'English'
    ) 
    TO '{output_file}' 
    (FORMAT JSON, COMPRESSION GZIP, ARRAY FALSE)
"""

con.execute(query)
print(f"\nSuccess! Filtered data has been saved to: {output_file}")