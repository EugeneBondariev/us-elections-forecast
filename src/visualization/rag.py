"""RAG system: embed district fact sheets, retrieve by natural language query."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from src.paths import DataFile

logger = logging.getLogger(__name__)

__all__ = ["build_index", "load_index", "retrieve", "rag_answer"]

# Override with GROQ_MODEL if Groq retires/renames this model.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def _district_to_text(row: dict | pd.Series) -> str:
    """Convert a district row to a searchable text document."""
    label = row.get("label", f"{row.get('state_po','?')}-{row.get('district','?')}")
    proba = row.get("proba_dem", 0.5)
    lean_str = "leans Democratic" if proba >= 0.6 else "leans Republican" if proba <= 0.4 else "is a toss-up"
    inc = str(row.get("district_incumbent", "unknown")).title()
    open_seat = "This is an open seat (no incumbent running)." if row.get("is_open", False) else f"The incumbent is {inc}."
    dem_lean_hist = row.get("dem_lean", None)
    hist_str = f"Historically, Democrats win about {dem_lean_hist*100:.1f}% of the two-party vote here." if dem_lean_hist and not np.isnan(float(dem_lean_hist)) else ""
    cook = row.get("cook_rating", None)
    cook_str = f"Cook Political Report rates this district: {cook}." if cook and str(cook) != "nan" else ""

    return (
        f"District {label}: The model gives Democrats a {proba*100:.1f}% win probability. "
        f"This district {lean_str}. {open_seat} {hist_str} {cook_str}"
    ).strip()


# chromadb imported lazily — downloads embedding model on first use
def build_index(districts: pd.DataFrame, persist: bool = True) -> object:
    """Build a ChromaDB collection from district fact sheets. Returns the collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(DataFile.CHROMA)) if persist else chromadb.EphemeralClient()

    # Delete existing collection to allow rebuild
    try:
        client.delete_collection("districts")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        "districts",
        metadata={"hnsw:space": "cosine"},
    )

    docs, ids, metas = [], [], []
    for row in districts.to_dict(orient="records"):
        text = _district_to_text(row)
        label = str(row.get("label", row.get("district", "unknown")))
        docs.append(text)
        ids.append(label)
        metas.append({
            "state": str(row.get("state_po", "")),
            "district": int(row.get("district", 0)),
            "proba_dem": float(row.get("proba_dem", 0.5)),
        })

    collection.add(documents=docs, ids=ids, metadatas=metas)
    logger.info("RAG index built: %d district documents", len(docs))
    return collection


def load_index() -> object:
    """Load persisted ChromaDB collection. Raises if not built yet."""
    import chromadb
    client = chromadb.PersistentClient(path=str(DataFile.CHROMA))
    return client.get_collection("districts")


def retrieve(query: str, n_results: int = 5, collection: object | None = None) -> list[dict]:
    """Retrieve the n most relevant district documents for a query."""
    if collection is None:
        collection = load_index()
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [{"text": d, **m} for d, m in zip(docs, metas)]


# groq imported lazily — optional dependency, only needed when GROQ_API_KEY is set
def rag_answer(query: str, collection: object | None = None, api_key: str | None = None) -> str:
    """Retrieve relevant districts and generate an LLM answer. Requires GROQ_API_KEY."""
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not set."

    chunks = retrieve(query, n_results=5, collection=collection)
    context = "\n".join(f"- {c['text']}" for c in chunks)

    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": "You are a nonpartisan election analyst. Answer questions about the 2026 US House forecast using only the provided district data. Be concise.",
            },
            {
                "role": "user",
                "content": f"Context (retrieved district data):\n{context}\n\nQuestion: {query}",
            },
        ],
    )
    return response.choices[0].message.content.strip()
