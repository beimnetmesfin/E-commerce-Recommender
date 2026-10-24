
import numpy as np
import pandas as pd

from app.faiss_service import FaissService


class Retriever:
    def __init__(self, faiss_service: FaissService):
        self.faiss_service = faiss_service

    def retrieve(
        self,
        user_profile: np.ndarray,
        products: pd.DataFrame,
        k: int = 50,
    ) -> pd.DataFrame:
        # 1. Validate input data
        if products.empty:
            raise ValueError("Products DataFrame is empty")

        if user_profile.ndim != 1:
            raise ValueError("User profile must be a 1D vector")

        if not np.isfinite(user_profile).all():
            raise ValueError("User profile must contain only finite values")

        if k < 1:
            raise ValueError("k must be at least 1")

        # 2. Validate the real FAISS index when available
        index = getattr(self.faiss_service, "index", None)

        if index is not None:
            if len(products) != index.ntotal:
                raise ValueError(
                    "Product count must match the number of FAISS index vectors"
                )

            if user_profile.shape[0] != index.d:
                raise ValueError(
                    "User profile dimension must match the FAISS index dimension"
                )

            k = min(k, index.ntotal)

        # 3. Search FAISS for similar products
        scores, indices = self.faiss_service.search(
            user_profile.astype(np.float32),
            k=k,
        )

        # 4. Validate search results
        scores = np.asarray(scores)
        indices = np.asarray(indices)

        if scores.ndim != 2 or indices.ndim != 2:
            raise ValueError("FAISS scores and indices must be 2D arrays")

        if scores.shape != indices.shape:
            raise ValueError("FAISS scores and indices must have matching shapes")

        if scores.shape[0] != 1:
            raise ValueError("FAISS search must return one result row per profile")

        scores = scores[0]
        indices = indices[0]

        # Remove invalid positions returned by FAISS.
        valid = (
            (indices >= 0)
            & (indices < len(products))
            & np.isfinite(scores)
        )
        indices = indices[valid]
        scores = scores[valid]

        # 5. Map vector positions to product rows
        candidates = products.iloc[indices].copy()
        candidates["similarity_score"] = scores

        # 6. Return candidates with a clean DataFrame index
        return candidates.reset_index(drop=True)
