import duckdb
import argparse
import sys
from sentence_transformers import SentenceTransformer
import numpy as np

# Database file
DB_FILE = "embeddings.db"

def get_db_connection():
    """Connect to DuckDB database in read-only mode."""
    conn = duckdb.connect(DB_FILE, read_only=True) 
    # Enable the vector extension (though array_cosine_similarity is built-in often)
    # However, for array functions we generally don't need extensions in recent versions.
    return conn

def main():
    parser = argparse.ArgumentParser(description="Find similar texts in the database.")
    parser.add_argument("-n", type=int, default=1, help="Number of nearest neighbors to find (default: 1)")
    args = parser.parse_args()
    
    n_neighbors = args.n
    
    print("Loading embedding model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Connecting to database...")
    try:
        conn = get_db_connection()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Please run prepare.py first.")
        sys.exit(1)
        
    print(f"Ready to search! (finding top {n_neighbors} matches)")
    print("Type your query and press Enter. Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            query = input("\nEnter query: ")
        except EOFError:
            break
            
        if query.lower() in ('exit', 'quit'):
            break
            
        if not query.strip():
            continue
            
        # Generates a vector (list of numbers) for the query text
        query_embedding = model.encode(query).tolist()
        
        # SQL query to find the nearest neighbors
        # We select relevant columns and calculate similarity
        # list_cosine_similarity computes the cosine similarity between two lists of numbers
        # The '?' are placeholders for our parameters (query_embedding and n_neighbors)
        # We order by similarity in descending order (highest similarity first)
        results = conn.execute("""
            SELECT 
                title, 
                author, 
                line_number, 
                filename, 
                content, 
                list_cosine_similarity(embedding, ?) as similarity
            FROM texts
            ORDER BY similarity DESC
            LIMIT ?
        """, [query_embedding, n_neighbors]).fetchall()
        
        # Display the results to the user
        print(f"\nFound {len(results)} matches:")
        for idx, row in enumerate(results):
            # Unpack the row tuple into variables
            title, author, line_num, filename, content, similarity = row
            print(f"\n--- Result {idx+1} (Similarity: {similarity:.4f}) ---")
            print(f"Source: {title} by {author} ({filename}:{line_num})")
            print(f"Text: {content}")

    conn.close()
    print("Goodbye!")

if __name__ == "__main__":
    main()
