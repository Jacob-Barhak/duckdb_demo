import duckdb
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Database file
DB_FILE = "embeddings.db"
TEXTS_DIR = "texts"

def get_db_connection():
    """Connect to DuckDB database."""
    conn = duckdb.connect(DB_FILE)
    return conn

def create_table(conn):
    """Create the table for storing texts and embeddings if it doesn't exist."""
    # Ensure the table is created with the PRIMARY KEY constraint on filename and line_number
    # This prevents duplicate entries for the same line in the same file.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS texts (
            line_number INTEGER,
            filename VARCHAR,
            author VARCHAR,
            title VARCHAR,
            content VARCHAR,
            embedding FLOAT[],
            PRIMARY KEY (filename, line_number)
        )
    """)
    # We do NOT clear existing data anymore, to allow incremental updates.
    print("Table 'texts' (re)created/verified.")

def parse_file(filepath):
    """
    Parse a text file to extract author, title, and content lines.
    
    Returns:
        tuple: (author, title, list_of_lines)
    """
    author = "Unknown"
    title = "Unknown"
    lines = [] # store (line_number, text)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        file_content = f.readlines()

    start_parsing = False
    
    for i, line in enumerate(file_content):
        line_num = i + 1  # 1-based index
        stripped_line = line.strip()
        
        # Extract metadata
        if stripped_line.startswith("Title: "):
            title = stripped_line.replace("Title: ", "").strip()
        elif stripped_line.startswith("Author: "):
            author = stripped_line.replace("Author: ", "").strip()
            
        # Check start/end markers
        # Be more flexible with markers
        upper_line = line.upper()
        if "START OF THE PROJECT GUTENBERG EBOOK" in upper_line:
            start_parsing = True
            continue
        if "END OF THE PROJECT GUTENBERG EBOOK" in upper_line:
            break
            
        if start_parsing:
            if stripped_line: # Only add non-empty lines
                lines.append((line_num, stripped_line))
                
    return author, title, lines

def main():
    # Initialize counters for reporting
    total_files_loaded = 0
    total_lines_processed = 0
    total_lines_inserted = 0
    total_lines_skipped = 0

    # Initialize the embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    conn = get_db_connection()
    create_table(conn)
    
    # Process each file in the texts directory
    if not os.path.exists(TEXTS_DIR):
        print(f"Directory '{TEXTS_DIR}' not found.")
        return

    files = [f for f in os.listdir(TEXTS_DIR) if f.endswith('.txt')]
    
    if not files:
        print(f"No text files found in '{TEXTS_DIR}'.")
        return

    for filename in files:
        filepath = os.path.join(TEXTS_DIR, filename)
        print(f"Processing file: {filename}")
        total_files_loaded += 1
        
        # Parse the file to get metadata and content lines
        author, title, content_lines = parse_file(filepath)
        print(f"  Title: {title}")
        print(f"  Author: {author}")
        print(f"  Lines to process: {len(content_lines)}")

        if not content_lines:
            print("  No content found to embed.")
            continue
            
        total_lines_processed += len(content_lines)

        # Separate line numbers and content for embedding generation
        line_numbers = [item[0] for item in content_lines]
        texts = [item[1] for item in content_lines]

        # Generate embeddings for all lines in batch.
        # This uses the SentenceTransformer model to convert text into a list of numbers (vector).
        print("  Generating embeddings...")
        embeddings = model.encode(texts)
        
        print("  Inserting into database...")
        # Prepare data for insertion
        # We need to restructure data as a list of tuples to insert into the database
        data_to_insert = []
        for i, text in enumerate(texts):
            line_num = line_numbers[i]
            # Convert numpy array to python list so DuckDB can handle it
            embedding_list = embeddings[i].tolist()
            data_to_insert.append((line_num, filename, author, title, text, embedding_list))
            
        # Check current count before insert
        count_before = conn.execute("SELECT count(*) FROM texts").fetchone()[0]
        
        # Bulk insert using INSERT OR IGNORE to skip duplicates
        # The '?' placeholders prevent SQL injection and handle data types correctly
        conn.executemany("""
            INSERT OR IGNORE INTO texts (line_number, filename, author, title, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, data_to_insert)
        
        # Check count after insert
        count_after = conn.execute("SELECT count(*) FROM texts").fetchone()[0]
        
        inserted_count = count_after - count_before
        skipped_count = len(data_to_insert) - inserted_count
        
        total_lines_inserted += inserted_count
        total_lines_skipped += skipped_count
        
        print(f"  Inserted: {inserted_count}, Skipped (Duplicate): {skipped_count}")

    conn.close()
    
    print("\nProcessing complete.")
    print(f"Total files processed: {total_files_loaded}")
    print(f"Total lines processed: {total_lines_processed}")
    print(f"Total lines inserted: {total_lines_inserted}")
    print(f"Total lines skipped (rejected): {total_lines_skipped}")

if __name__ == "__main__":
    main()
