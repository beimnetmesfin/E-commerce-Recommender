from pathlib import Path
import numpy as np

#Import actual class
from app.config import (item_embeddings_file,
                        faiss_index_file,
                        embedding_dimension,)
from app.data_loader import DataLoader
from app.user_profile_vector import UserProfileBuilder
from app.faiss_service import FaissService
from app.retrieval import Retriever
from app.business_rule import BusinessRules
from app.ranker import Ranker
from app.recommender import Recommender

#Load saved embeddings
item_embeddings=np.load(item_embeddings_file)
data_loader=DataLoader(products_path=Path("data/products.csv"),
                       user_events_path=Path("data/user_events.csv"))
profile_builder=UserProfileBuilder(embedding_dimension=embedding_dimension)
faiss_service=FaissService(dimension=embedding_dimension)
faiss_service.load_index(faiss_index_file)
retriver=Retriever(faiss_service=faiss_service)
business_rule=BusinessRules()
ranker=Ranker(top_k=10,max_per_category=3)

recommender=Recommender(
    load_products=DataLoader.load_products,
    load_events=DataLoader.load_user_events,

    build_profile=lambda user_events,user_id:
    profile_builder.build_profile(
        user_events=user_events,
        item_embeddings=item_embeddings,
        user_id=user_id
    ),

    retrieve_candidates=lambda profile,products:
    retriver.retrieve(user_profile=profile,products=products,k=50),

    apply_business_rules=lambda candidates,user_id,user_events:
    business_rule.apply_rules(
        candidates=candidates,
        user_id=user_id,
        user_events=user_events),
    ranker=ranker
)

recommendations=recommender.recommend_for_user(user_id=1,top_k=10)
print(recommendations)

