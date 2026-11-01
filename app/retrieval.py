
from pathlib import Path

import numpy as np
import pandas as pd

from app.faiss_service import FaissService


class Retriever:

    def __init__(
        self,
        faiss_service: FaissService,
        item_ids_file: Path,
    ):
        self.faiss_service = faiss_service

        # Load the product IDs in FAISS vector order
        self.item_ids = np.load(item_ids_file, allow_pickle=False)

        # Validate the ID mapping
        if self.item_ids.ndim != 1:
            raise ValueError(
                "Product IDs must be a 1-dimensional array."
            )

        if len(np.unique(self.item_ids)) != len(self.item_ids):
            raise ValueError(
                "Product ID mapping contains duplicate IDs."
            )

        # Build a fast product ID lookup
        self.item_id_to_position = {
            item_id: position
            for position, item_id in enumerate(self.item_ids)
        }

    def retrieve(
        self,
        user_profile: np.ndarray,
        products: pd.DataFrame,
        k: int = 50,
    ) -> pd.DataFrame:

        # 1. Validate product data
        if products.empty:
            raise ValueError("Products DataFrame is empty")

        if "item_id" not in products.columns:
            raise ValueError(
                "Products DataFrame must contain item_id."
            )

        if products["item_id"].isna().any():
            raise ValueError(
                "Products DataFrame contains missing item IDs."
            )

        if products["item_id"].duplicated().any():
            raise ValueError(
                "Products DataFrame contains duplicate item IDs."
            )

        # 2. Validate user profile
        user_profile = np.asarray(user_profile)

        if user_profile.ndim != 1:
            raise ValueError(
                "User profile must be a 1D vector."
            )

        if not np.isfinite(user_profile).all():
            raise ValueError(
                "User profile must contain only finite values."
            )

        if k < 1:
            raise ValueError("k must be at least 1")

        # 3. Validate FAISS index
        index = self.faiss_service.index

        if index.ntotal != len(self.item_ids):
            raise ValueError(
                "FAISS index count does not match product ID mapping."
            )

        if user_profile.shape[0] != index.d:
            raise ValueError(
                "User profile dimension does not match FAISS index."
            )

        if not np.isfinite(user_profile).all():
            raise ValueError(
                "User profile contains invalid numerical values."
            )

        # 4. Validate product ID availability
        products = products.copy()

        products["item_id"] = products["item_id"].astype(str)

        missing_ids = set(self.item_ids) - set(products["item_id"])

        if missing_ids:
            raise ValueError(
                f"{len(missing_ids)} indexed products are missing "
                "from the current product data."
            )

        # 5. Search FAISS
        k = min(k, index.ntotal)

        scores, indices = self.faiss_service.search(
            user_profile.astype(np.float32),
            k=k,
        )

        # 6. Validate search results
        scores = np.asarray(scores)
        indices = np.asarray(indices)

        if scores.ndim != 2 or indices.ndim != 2:
            raise ValueError(
                "FAISS scores and indices must be 2D arrays."
            )

        if scores.shape != indices.shape:
            raise ValueError(
                "FAISS scores and indices must have matching shapes."
            )

        if scores.shape[0] != 1:
            raise ValueError(
                "FAISS search must return one result row per profile."
            )

        scores = scores[0]
        indices = indices[0]

        # 7. Remove invalid FAISS positions
        valid = (
            (indices >= 0)
            & (indices < len(self.item_ids))
            & np.isfinite(scores)
        )

        indices = indices[valid]
        scores = scores[valid]

        # 8. Convert FAISS positions into product IDs
        retrieved_ids = self.item_ids[indices].astype(str)

        # 9. Retrieve matching product records by item_id
        products_by_id = products.set_index(
            "item_id",
            drop=False
        )

        candidates = products_by_id.loc[
            retrieved_ids
        ].copy()

        # 10. Attach similarity scores
        candidates["similarity_score"] = scores

        # 11. Return candidates
        return candidates.reset_index(drop=True)
