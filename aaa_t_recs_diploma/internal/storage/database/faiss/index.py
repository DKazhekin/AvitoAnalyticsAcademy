import faiss


def create_faiss_index(embedding_dim) -> faiss.IndexFlatIP:
    return faiss.IndexIDMap2(faiss.IndexFlatIP(embedding_dim))
