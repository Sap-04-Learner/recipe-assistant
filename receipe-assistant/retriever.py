import chromadb
from chromadb.utils import embedding_functions

from config import DB_PATH, COLLECTION_NAME, OLLAMA_URL, EMBED_MODEL


def get_recipe_recommendations(user_ingredients, top_k=3):
    """
    Takes a string of ingredients, converts it to a vector, 
    and searches ChromaDB for the closest recipe matches.
    """
    # 1. Connect to our existing database
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 2. Set up the exact same embedding model we used to index
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL
    )
    
    # 3. Get the collection we just built
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef  # type: ignore
    )
    
    # 4. Perform the semantic search
    results = collection.query(
        query_texts=[user_ingredients],
        n_results=top_k
    )
    
    # ChromaDB returns a nested dictionary. We just want the list of recipe metadata.
    if results['metadatas']:
        return results['metadatas'][0]
    else:
        return []

# --- Quick Test Block ---
# This block only runs if you execute this specific file directly.
if __name__ == "__main__":
    test_query = "chicken, broccoli, soy sauce"
    print(f"Searching database for: {test_query}\n")
    
    matches = get_recipe_recommendations(test_query, top_k=3)
    
    if not matches:
        print("No matches found.")
    else:
        for i, match in enumerate(matches):
            print(f"--- Match {i+1} ---")
            print(f"Name: {match.get('name')}")
            print(f"Time: {match.get('minutes')} mins")
            print(f"Ingredients: {match.get('ingredients')}\n")