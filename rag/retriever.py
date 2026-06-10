import os

from rag.vector_store import (
    load_vector_store
)

def get_retriever():

    if not os.path.exists(
        "vectorstore/index.faiss"
    ):
        return None

    vectorstore = load_vector_store()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )