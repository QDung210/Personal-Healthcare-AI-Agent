"""
Test Qdrant connection 
"""

import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv() 

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

def test_qdrant_connection():
    print("🔍 Testing Qdrant connection...\n")

    if not QDRANT_URL or not QDRANT_API_KEY:
        print("Missing Qdrant configuration in .env file.")
        print("   → Please set QDRANT_URL and QDRANT_API_KEY.")
        return

    try:
        # Create client
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT,
        )

        # Test 1: List collections
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        print("Qdrant connection successful!")
        print(f"Collections: {collection_names}\n")

        #  Test 2: Check if target collection exists
        if QDRANT_COLLECTION_NAME in collection_names:
            info = client.get_collection(QDRANT_COLLECTION_NAME)
            print(f"'{QDRANT_COLLECTION_NAME}' found!")
            print(f"   Vector count: {info.vectors_count}")
        else:
            print(f"Collection '{QDRANT_COLLECTION_NAME}' not found.")

    except Exception as e:
        print(f"Qdrant connection failed: {e}")
        print("\nPossible fixes:")
        print("   1. Check internet connection")
        print("   2. Verify Qdrant URL in .env")
        print("   3. Ensure API key is valid")
        print("   4. Check if your collection name exists")

if __name__ == "__main__":
    test_qdrant_connection()
