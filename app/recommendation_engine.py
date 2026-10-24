
import numpy as np

from app.config import (
    products_file,
    user_events_file,
    item_embeddings_file,
    faiss_index_file,
    embedding_dimension,
    retrival_number,
    top_recommendation,
)
from app.data_loader import DataLoader
from app.faiss_service import FaissService
from app.retrieval import Retriever
from app.user_profile_vector import UserProfileBuilder
from app.business_rule import BusinessRules
from app.ranker import Ranker
from app.recommender import Recommender
from api.event_database import load_events_dataframe


# 1. Load the products.
data_loader = DataLoader(products_file, user_events_file)
products = data_loader.load_products()

# 2. Load the saved item embeddings.
item_embeddings = np.load(item_embeddings_file)

if len(products) != len(item_embeddings):
    raise ValueError(
        "Product count does not match embedding row count"
    )

# 3. Create the mapping from item ID to embedding row.
item_id_to_index = {
    int(item_id): index
    for index, item_id in enumerate(products["item_id"])
}

if len(item_id_to_index) != len(products):
    raise ValueError("Product item IDs must be unique")

# 4. Load the saved FAISS index.
faiss_service = FaissService(dimension=embedding_dimension)
faiss_service.load_index(faiss_index_file)

if faiss_service.index.ntotal != len(products):
    raise ValueError(
        "Product count does not match FAISS vector count"
    )

# 5. Initialize the components.
profile_builder = UserProfileBuilder(
    embedding_dimension=embedding_dimension
)

retriever = Retriever(faiss_service)
business_rules = BusinessRules()
ranker = Ranker(top_k=top_recommendation)

# 6. Connect all components.
recommender = Recommender(
    load_products=lambda: data_loader.load_products(),
    load_events=load_events_dataframe,
    item_embeddings=item_embeddings,

    build_profile=lambda user_events, embeddings, user_id: (
        profile_builder.build_profile(
            user_events=user_events,
            item_embeddings=embeddings,
            user_id=user_id,
            item_id_to_index=item_id_to_index,
        )
    ),

    retrieve_candidates=lambda profile, product_df: (
        retriever.retrieve(
            user_profile=profile,
            products=product_df,
            k=retrival_number,
        )
    ),

    apply_business_rules=lambda candidates, user_id, events: (
        business_rules.apply_rules(
            candidates=candidates,
            user_id=user_id,
            user_events=events,
        )
    ),

    ranker=ranker,
)


def recommend_for_user(user_id, top_k=10):
    """Generate ranked recommendations for a user."""
    return recommender.recommend_for_user(
        user_id=user_id,
        top_k=top_k,
    )
