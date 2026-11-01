
import numpy as np

from app.config import (
    products_file,
    user_events_file,
    item_embeddings_file,
    item_ids_file,
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


# --------------------------------------------------
# 1. Load product and event data
# --------------------------------------------------

data_loader = DataLoader(
    products_file,
    user_events_file,
)

products = data_loader.load_products()


# --------------------------------------------------
# 2. Load saved embeddings and product ID mapping
# --------------------------------------------------

item_embeddings = np.load(
    item_embeddings_file,
    allow_pickle=False,
)

item_ids = np.load(
    item_ids_file,
    allow_pickle=False,
)

if item_embeddings.ndim != 2:
    raise ValueError("Item embeddings must be a 2D matrix")

if item_embeddings.shape[1] != embedding_dimension:
    raise ValueError("Embedding dimension does not match configuration")

if len(item_embeddings) != len(item_ids):
    raise ValueError(
        "Embedding count does not match product ID mapping"
    )

if len(products) == 0:
    raise ValueError("Product dataset is empty")

if products["item_id"].isna().any():
    raise ValueError("Product dataset contains missing item IDs")

if products["item_id"].astype(str).duplicated().any():
    raise ValueError("Product IDs must be unique")


# --------------------------------------------------
# 3. Create item ID to embedding position mapping
# --------------------------------------------------

item_id_to_index = {
    int(item_id): position
    for position, item_id in enumerate(item_ids)
}

if len(item_id_to_index) != len(item_ids):
    raise ValueError("Product ID mapping contains duplicate IDs")


# --------------------------------------------------
# 4. Load saved FAISS index
# --------------------------------------------------

faiss_service = FaissService(
    dimension=embedding_dimension,
)

faiss_service.load_index(faiss_index_file)

if faiss_service.index.ntotal != len(item_ids):
    raise ValueError(
        "FAISS index count does not match saved product IDs"
    )


# --------------------------------------------------
# 5. Initialize recommendation components
# --------------------------------------------------

profile_builder = UserProfileBuilder(
    embedding_dimension=embedding_dimension,
)

retriever = Retriever(
    faiss_service=faiss_service,
    item_ids_file=item_ids_file,
)

business_rules = BusinessRules()

ranker = Ranker(
    top_k=top_recommendation,
)


# --------------------------------------------------
# 6. Connect all components
# --------------------------------------------------

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

    item_ids=item_ids,
)


# --------------------------------------------------
# 7. Public recommendation functions
# --------------------------------------------------

def recommend_for_user(user_id, top_k=10):
    """Generate personalized recommendations."""
    return recommender.recommend_for_user(
        user_id=user_id,
        top_k=top_k,
    )


def get_similar_products(item_id, top_k=4):
    """Find products similar to a selected product."""
    return recommender.get_similar_products(
        item_id=item_id,
        top_k=top_k,
    )
