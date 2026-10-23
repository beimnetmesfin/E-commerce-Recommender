from typing import Dict
import pandas as pd
import numpy as np

# The user profile vector is the important because 
# it converts the user behavoiur into the quary vector to search in FAISS
class UserProfileBuilder:
    event_weights:Dict[str,float]={
        "view":1.0,"click":2.0,"like":3.0,"add_to_cart":4.0,"skip":0.5
        }
    def __init__(self,embedding_dimension:int):
        self.embedding_dimnesion=embedding_dimension

    def build_profile(self,
            user_events:pd.DataFrame,
            item_embeddings:np.ndarray,
            user_id:int)->np.ndarray:
        #Item_embeddings is the item_embeddings of products that the user interacted with in websites
        user_events=user_events[user_events["user_id"]==user_id].copy()
        if user_events.empty:
            raise ValueError(f"No events found for user :{user_id}")
        if item_embeddings.ndim!=2:
            raise ValueError("Item embeddings must be 2D matrix")
        if item_embeddings.shape[1]!=self.embedding_dimnesion:
            raise ValueError("Embedding dimension does not match")

        user_events["event_weight"]=user_events["event_type"].map(self.event_weights)
        if user_events["event_weight"].isna().any():
            unknown_events=(
                user_events.loc[
                    user_events["event_weight"].isna(),
                    "event_type"
                ].unique().tolist()
            )
            raise ValueError(
                f"Unknown event types :{unknown_events}")
        user_events["recency_weight"]=(1.0/user_events["recency_rank"])
        user_events["final_weight"]=(
            user_events["event_weight"]*user_events["recency_weight"]
        )

        weighted_vectors=[]
        weights=[]
        for _,event in user_events.iterrows():          
            item_index=int(event["item_id"])
            vector=item_embeddings[item_index]
            weighted_vectors.append(
                vector*event["final_weight"]
            )
            weights.append(event["final_weight"])
        weighted_sum=np.sum(weighted_vectors,axis=0)
        total_weights=np.sum(weights)
        if total_weights<=0:                        
            raise ValueError("Total profile weight must be greater than zero")
        profile_vector=(weighted_sum/total_weights)
        norm=np.linalg.norm(profile_vector)
        if norm==0:    
            raise ValueError("User profile vector has mangnitude")
        profile_vector=(profile_vector/norm).astype(np.float32)
        return profile_vector
                    


