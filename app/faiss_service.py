from pathlib import Path
import numpy as np
import faiss
#FAISS (Facebook AI Similarity Search) is an open-source library by Meta
#  that stores high-dimensional vectors and finds similar items fast.
class FaissService:
    #FAISS index is the fast vector-search structure.
    def __init__(self,dimension:int):
        self.dimension=dimension
        self.index=faiss.IndexFlatIP(dimension)#IndexFlatIP performs exact inner-product search
    def build_index(self,embedding:np.ndarray)->None:
        embeddings=embedding.astype("float32").copy()
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    def search(self,query_embedding:np.ndarray,k:int=10):
        query_embedding=query_embedding.astype("float32").copy()
        if query_embedding.ndim==1:
            query_embedding=query_embedding(1,-1)
        faiss.normalize_L2(query_embedding)
        scores,indices=self.index.search(query_embedding,k)
        return scores,indices
    def save_index(self,path:Path)->None:
        path=Path(path)
        path.parent.mkdir(parents=True,
                          exist_ok=True
                          )
        faiss.write_index(
            self.index,
            str(path)
        )
    def load_index(self,path:Path)->None:
        path=Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found:{path}"
            )
        self.index=faiss.read_index(str(path))