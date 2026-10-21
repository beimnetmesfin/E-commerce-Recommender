from typing import Optional
import numpy as np 
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self,embedding_model_name:str,device:Optional[str]=None):
        self.model=SentenceTransformer(
          embedding_model_name,
            device=device
        )
    #Create Embeddings text
    def encode(self,texts:list[str],batch_size:int=32)->np.ndarray:
        if not texts:
            raise ValueError("text must contain at least one item.")
        embedding=self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype("float32")
    @property
    def dimension(self)->int:
        #Return the embedding dimension of the loaded model
        return self.model.get_embedding_dimension()
    