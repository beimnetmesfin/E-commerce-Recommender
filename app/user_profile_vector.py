
from typing import Dict, Optional

import numpy as np
import pandas as pd

class UserProfileBuilder:
    event_weights: Dict[str, float] = {
        "view": 1.0,
        "click": 2.0,
        "like": 3.0,
        "add_to_cart": 4.0,
        "purchase": 5.0,
        "skip": 0.5,
    }

    def __init__(self, embedding_dimension: int):
        self.embedding_dimension = embedding_dimension

    def build_profile(
        self,
        user_events: pd.DataFrame,
        item_embeddings: np.ndarray,
        user_id: str | int,
        item_id_to_index: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        if user_events.empty:
            raise ValueError(f"No events found for user: {user_id}")

        if item_embeddings.ndim != 2:
            raise ValueError("Item embeddings must be a 2D matrix")

        if item_embeddings.shape[1] != self.embedding_dimension:
            raise ValueError("Embedding dimension does not match")

        required_columns = {"item_id", "event_type", "recency_rank"}
        missing = required_columns - set(user_events.columns)

        if missing:
            raise ValueError(
                f"User events are missing required columns: {sorted(missing)}"
            )

        events = user_events.copy()

        # Convert event types into preference weights.
        events["event_type"] = (
            events["event_type"].astype(str).str.strip().str.lower()
        )
        events["event_weight"] = events["event_type"].map(self.event_weights)

        if events["event_weight"].isna().any():
            unknown_events = events.loc[
                events["event_weight"].isna(), "event_type"
            ].unique().tolist()
            raise ValueError(f"Unknown event types: {unknown_events}")

        # Ensure item IDs are numeric for the mapping.
        events["item_id"] = pd.to_numeric(
            events["item_id"], errors="raise"
        ).astype(int)

        # Map each item ID to its actual embedding row.
        if item_id_to_index is not None:
            events["item_index"] = events["item_id"].map(item_id_to_index)

            if events["item_index"].isna().any():
                missing_ids = events.loc[
                    events["item_index"].isna(), "item_id"
                ].unique().tolist()
                raise ValueError(
                    f"No embedding row mapping found for item IDs: {missing_ids}"
                )

            events["item_index"] = events["item_index"].astype(int)
        else:
            # Default supports the zero-based IDs used in the unit tests.
            events["item_index"] = events["item_id"]

        if (
            (events["item_index"] < 0).any()
            or (events["item_index"] >= len(item_embeddings)).any()
        ):
            raise ValueError(
                "An item ID does not map to a valid embedding row. "
                "Provide item_id_to_index for non-zero-based IDs."
            )

        # Rank 0 means most recent; adding 1 prevents division by zero.
        events["recency_rank"] = pd.to_numeric(
            events["recency_rank"], errors="coerce"
        ).fillna(0)

        events["recency_weight"] = 1.0 / (
            events["recency_rank"].clip(lower=0) + 1.0
        )

        events["final_weight"] = (
            events["event_weight"] * events["recency_weight"]
        )

        indices = events["item_index"].to_numpy(dtype=int)
        weights = events["final_weight"].to_numpy(dtype=np.float32)

        # Compute the weighted average of the interacted item embeddings.
        weighted_sum = np.sum(
            item_embeddings[indices] * weights[:, None],
            axis=0,
        )
        total_weight = float(weights.sum())

        if total_weight <= 0 or not np.isfinite(total_weight):
            raise ValueError("Total profile weight must be positive and finite")

        profile_vector = weighted_sum / total_weight
        norm = np.linalg.norm(profile_vector)

        if norm == 0 or not np.isfinite(norm):
            raise ValueError("User profile vector has an invalid magnitude")

        # Normalize the final profile for cosine-similarity retrieval.
        return (profile_vector / norm).astype(np.float32)
