"""Build the ChromaDB RAG index from the current forecast."""

from src.models.forecast import run_forecast
from src.visualization.rag import build_index, retrieve


def main() -> None:
    print("Running forecast to get districts...")
    result = run_forecast()
    districts = result["districts"]
    print(f"Building RAG index for {len(districts)} districts...")
    collection = build_index(districts, persist=True)
    print("Index built. Stored at data/chroma_db/")

    # Quick smoke test
    hits = retrieve("competitive swing district Pennsylvania", n_results=3, collection=collection)
    print("\nSample retrieval — 'competitive swing district Pennsylvania':")
    for h in hits:
        print(f"  {h['state']}-{h['district']}: {h['text'][:100]}...")


if __name__ == "__main__":
    main()
