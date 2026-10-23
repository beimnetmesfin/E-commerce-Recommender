import numpy as np
import pandas as pd
from app.faiss_service import FaissService

class Retriever:
    def __init__(self,faiss_service:FaissService):
        self.faiss_service=faiss_service
    def retrieve(self,
                user_profile:np.ndarrary,
                products:pd.DataFrame,
                k:int=50
                )->pd.DataFrame:
        if products.empty:
            raise ValueError("Products Dataframe is empty")
        if user_profile.ndim!=1:
            raise ValueError("User profile must be a 1D vector")
        scores,indices=self.faiss_service.search(user_profile,k=k)#Search using faiss
        scores=scores[0]
        indices=indices[0]
        candidates=products.iloc[indices].copy()
        candidates["similarities"]=scores
        candidates=candidates.reset_index(drop=True)
        return candidates

        

        
    