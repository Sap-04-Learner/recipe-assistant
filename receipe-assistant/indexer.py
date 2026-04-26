import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import sys

# Importing directly from your config file
from config import CSV_PATH, DB_PATH, COLLECTION_NAME, OLLAMA_URL, EMBED_MODEL, BATCH_SIZE


def build_vector_db():
    print("1. Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)

    print(f"2. Connecting to local Ollama ({EMBED_MODEL})...")
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL
    )

    print(f"3. Preparing Collection: '{COLLECTION_NAME}'...")
    # Safe delete and create to prevent duplication/errors on re-runs
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("   - Existing collection wiped.")
    except Exception:
        pass # Collection didn't exist yet, which is fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef  # type: ignore
    )

    print(f"4. Loading dataset from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"\n[ERROR] File not found at {CSV_PATH}.")
        sys.exit(1)

    # CRITICAL FIX 1: Strip all NaN/Null values from the dataframe. 
    # ChromaDB will crash if a metadata value is NaN.
    df = df.fillna("") 
    df = df.reset_index(drop=True)

    print("5. Formatting data for ChromaDB...")
    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        # CRITICAL FIX 2: Ensure no empty strings are passed as documents
        doc_text = str(row.get('text_to_embed', '')).strip()
        if not doc_text:
            continue

        documents.append(doc_text)
        
        # CRITICAL FIX 3: ChromaDB metadata strictly rejects NumPy types.
        # Everything must be explicitly cast to standard Python str, int, or float.
        metadatas.append({
            "name": str(row.get('name', 'Unknown')),
            "minutes": int(row.get('minutes', 0)),
            "ingredients": str(row.get('clean_ingredients', '')),
            "steps": str(row.get('clean_steps', ''))
        })
        
        ids.append(f"recipe_{row.get('id', index)}")

    if not documents:
        print("\n[ERROR] No valid documents to embed. Check your CSV.")
        sys.exit(1)

    print(f"6. Generating vectors and indexing (Batch Size: {BATCH_SIZE})...")
    total_docs = len(documents)
    
    # Safe batching logic
    for i in tqdm(range(0, total_docs, BATCH_SIZE), desc="Indexing Progress"):
        end_idx = min(i + BATCH_SIZE, total_docs)
        
        # Slice the lists for this specific batch
        batch_docs = documents[i:end_idx]
        batch_metadatas = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]
        
        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        except Exception as e:
            print(f"\n[ERROR] ChromaDB failed at batch {i}-{end_idx}. Error Details:\n{e}")
            sys.exit(1)

    print(f"\n[SUCCESS] Vector database completely built with {total_docs} recipes!")

if __name__ == "__main__":
    build_vector_db()