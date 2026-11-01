
import numpy as np

from app.config import (
    item_embeddings_file,
    faiss_index_file,
    embedding_dimension,
    models_dir,
)

from app.faiss_service import FaissService


def main() -> None:

    # 1. Load product embeddings
    print("Loading embeddings...")

    embeddings = np.load(item_embeddings_file)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    # 2. Load product ID mapping
    print("Loading product IDs...")

    item_ids_file = models_dir / "item_ids.npy"
    item_ids = np.load(item_ids_file)

    print(f"Product IDs shape: {item_ids.shape}")

    # 3. Validate embedding structure
    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2-dimensional matrix."
        )

    if embeddings.shape[1] != embedding_dimension:
        raise ValueError(
            f"Expected embedding dimension: {embedding_dimension}, "
            f"but got {embeddings.shape[1]}."
        )

    # 4. Validate product ID structure
    if item_ids.ndim != 1:
        raise ValueError(
            "Product IDs must be a 1-dimensional array."
        )

    if len(item_ids) != len(embeddings):
        raise ValueError(
            "Number of product IDs does not match number of embeddings."
        )

    if len(np.unique(item_ids)) != len(item_ids):
        raise ValueError(
            "Product IDs contain duplicates."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain invalid numerical values."
        )

    # 5. Prepare embeddings for FAISS
    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    # 6. Create FAISS service
    print("Creating FAISS index...")

    faiss_service = FaissService(
        dimension=embedding_dimension
    )

    # 7. Add embeddings to FAISS
    print("Adding embeddings to FAISS index...")

    faiss_service.build_index(embeddings)

    # 8. Validate the constructed index
    if faiss_service.index.ntotal != len(item_ids):
        raise ValueError(
            "FAISS index vector count does not match product ID count."
        )

    if faiss_service.index.d != embedding_dimension:
        raise ValueError(
            "FAISS index dimension does not match expected dimension."
        )

    print(f"Number of vectors: {faiss_service.index.ntotal}")
    print(f"Index dimension: {faiss_service.index.d}")

    # 9. Save FAISS index
    print("Saving FAISS index...")

    faiss_service.save_index(faiss_index_file)

    print(f"FAISS index saved to: {faiss_index_file}")
    print("FAISS index built successfully.")


if __name__ == "__main__":
    main()
