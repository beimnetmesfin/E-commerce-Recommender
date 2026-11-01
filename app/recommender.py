
import pandas as pd
import numpy as np


class Recommender:
    """Connect all recommendation components into one pipeline."""

    def __init__(
        self,
        load_products,
        load_events,
        item_embeddings,
        build_profile,
        retrieve_candidates,
        apply_business_rules,
        ranker,
        item_ids,
    ):
        self.load_products = load_products
        self.load_events = load_events
        self.item_embeddings = np.asarray(
            item_embeddings,
            dtype=np.float32,
        )
        self.build_profile = build_profile
        self.retrieve_candidates = retrieve_candidates
        self.apply_business_rules = apply_business_rules
        self.ranker = ranker

        self.item_ids = np.asarray(item_ids).astype(str)

        if self.item_ids.ndim != 1:
            raise ValueError("item_ids must be a 1-dimensional array")

        if self.item_embeddings.ndim != 2:
            raise ValueError("item_embeddings must be a 2-dimensional array")

        if len(self.item_ids) != len(self.item_embeddings):
            raise ValueError(
                "item_ids and item_embeddings must have the same length"
            )

        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("item_ids must be unique")

        self.item_id_to_position = {
            item_id: position
            for position, item_id in enumerate(self.item_ids)
        }

    def recommend_for_user(self, user_id, top_k=10):
        """Generate personalized recommendations for a user."""

        if user_id is None or pd.isna(user_id):
            raise ValueError("user_id must not be empty")

        user_id = str(user_id).strip()

        if not user_id:
            raise ValueError("user_id must not be empty")

        if top_k is not None and (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer or None")

        products = self.load_products()
        events = self.load_events()

        if products is None or products.empty:
            return pd.DataFrame()

        if events is None or events.empty:
            return pd.DataFrame()

        user_events = events[
            events["user_id"].astype(str).str.strip() == user_id
        ].copy()

        if user_events.empty:
            return pd.DataFrame()

        profile = self.build_profile(
            user_events,
            self.item_embeddings,
            user_id,
        )

        if profile is None:
            return pd.DataFrame()

        candidates = self.retrieve_candidates(profile, products)

        if candidates is None or candidates.empty:
            return pd.DataFrame()

        eligible = self.apply_business_rules(
            candidates,
            user_id,
            events,
        )

        if eligible is None or eligible.empty:
            return pd.DataFrame()

        return self.ranker.rank(eligible, top_k=top_k)

    def get_similar_products(self, item_id, top_k=10):
        """Find products similar to a selected product."""

        if top_k is not None and (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer or None")

        item_id = str(item_id).strip()

        if not item_id:
            raise ValueError("item_id must not be empty")

        if item_id not in self.item_id_to_position:
            raise ValueError(f"Unknown item_id: {item_id}")

        products = self.load_products()

        if products is None or products.empty:
            return pd.DataFrame()

        position = self.item_id_to_position[item_id]

        query_embedding = self.item_embeddings[position]

        candidates = self.retrieve_candidates(
            query_embedding,
            products,
        )

        if candidates is None or candidates.empty:
            return pd.DataFrame()

        candidates = candidates[
            candidates["item_id"].astype(str) != item_id
        ].copy()

        if candidates.empty:
            return pd.DataFrame()

        eligible = self.apply_business_rules(
            candidates,
            None,
            pd.DataFrame(),
        )

        if eligible is None or eligible.empty:
            return pd.DataFrame()

        return self.ranker.rank(eligible, top_k=top_k)
