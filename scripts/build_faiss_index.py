import numpy as np
from app.config import (
    item_embeddings_file,
    faiss_index_file,
    embedding_dimension
    )
from app.faiss_service import FaissService
def main():
    print("Loading the embeddings file....")
    embeddings=np.load(item_embeddings_file)
    print(f"Embedding shape:{embeddings.shape} ")
    if embeddings.ndim!=2:
        raise ValueError(
            "Embeddings must be a 2-dimensional matrix"
        )
    if embeddings.shape[1]!=embedding_dimension:
        raise ValueError(
            f"Expected embedding dimension is :{embedding_dimension} "
            f"but got {embeddings.shape[1]}"
        )
    #creating the faiss in index
    faiss_service=FaissService(dimension=embedding_dimension)
    #Adding the embeddings in the FAISS Index
    faiss_service.build_index(embeddings)
    print(f"Number of vectors in Index....{faiss_service.index.ntotal}")
    print("Saving FAISS index....")
    faiss_service.save_index(faiss_index_file)
    print(f"FAISS index saved to :{faiss_index_file}")
    print(f"FAISS index build completed succesfully")
if __name__=="__main__":
    main()
