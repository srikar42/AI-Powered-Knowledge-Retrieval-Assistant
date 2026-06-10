from langchain_community.vectorstores import FAISS

from rag.embeddings import (
    get_embeddings
)

import shutil
import os


def create_vector_store(chunks):

    embeddings = get_embeddings()

    # Delete old vector store
    if os.path.exists("vectorstore"):

        shutil.rmtree(
            "vectorstore"
        )

    # Create fresh FAISS
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(
        "vectorstore"
    )

    return vectorstore


def load_vector_store():

    embeddings = get_embeddings()

    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )