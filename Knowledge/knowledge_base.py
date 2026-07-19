"""Dummy news knowledge base backed by a persisted Chroma vector store.

`build_news_vector_db()` returns a langchain-chroma `Chroma` instance that
exposes `.similarity_search(query, k=...)` returning documents with a
`.page_content` attribute — exactly the interface RagAgent depends on.
"""

import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "dummy_news"

# Dummy news articles used to seed the vector store.
DUMMY_NEWS = [
    "Global markets rallied today as the central bank held interest rates steady, "
    "with technology stocks leading the gains.",
    "Scientists announced a breakthrough in solid-state battery technology that could "
    "double the range of electric vehicles within five years.",
    "The national football team secured a dramatic last-minute win in the championship "
    "final, ending a decade-long title drought.",
    "A new climate report warns that coastal cities must accelerate flood-defense "
    "investment as sea levels rise faster than previously projected.",
    "The latest smartphone from a leading manufacturer features an on-device AI "
    "assistant and a three-day battery, launching next month.",
    "Health officials reported a sharp decline in seasonal flu cases this year, "
    "crediting an earlier and wider vaccination campaign.",
    "A major studio's space epic broke opening-weekend box office records, grossing "
    "over 300 million dollars worldwide.",
    "Regulators approved the merger of two regional airlines, promising more direct "
    "routes but raising concerns about ticket prices.",
]


def build_news_vector_db(persist_directory=CHROMA_DIR):
    """Return a Chroma vector store seeded with DUMMY_NEWS.

    The store is persisted to `persist_directory`; on subsequent runs the
    existing collection is reused instead of re-embedding the documents.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    already_seeded = os.path.isdir(persist_directory) and vector_db._collection.count() > 0
    if not already_seeded:
        docs = [
            Document(page_content=text, metadata={"source": "dummy_news", "id": i})
            for i, text in enumerate(DUMMY_NEWS)
        ]
        vector_db.add_documents(docs)

    return vector_db
