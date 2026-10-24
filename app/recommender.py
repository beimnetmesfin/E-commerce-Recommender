
import pandas as pd


class Recommender:
    """Connect all existing components into one recommendation pipeline."""

    def __init__(
        self,
        load_products,
        load_events,
        item_embeddings,
        build_profile,
        retrieve_candidates,
        apply_business_rules,
        ranker,
    ):
        self.load_products = load_products
        self.load_events = load_events
        self.item_embeddings = item_embeddings
        self.build_profile = build_profile
        self.retrieve_candidates = retrieve_candidates
        self.apply_business_rules = apply_business_rules
        self.ranker = ranker

    def recommend_for_user(self, user_id, top_k=10):
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

        if events.empty:
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

        if top_k is None:
            return self.ranker.rank(eligible)

        return self.ranker.rank(eligible, top_k=top_k)
