
import numpy as np

from app.config import (
    products_file,
    user_events_file,
    embedding_model_name,
    models_dir,
    item_embeddings_file,
)

from app.data_loader import DataLoader
from app.text_builder import build_text_embedding
from app.embedding_service import EmbeddingService


def main() -> None:

    # 1. Load product data
    print("Loading product data...")

    loader = DataLoader(
        products_file,
        user_events_file
    )

    products_df = loader.load_products()

    print(f"Loaded {len(products_df)} products.")

    # 2. Validate product IDs
    print("Validating product IDs...")

    if products_df.empty:
        raise ValueError("Product DataFrame is empty.")

    if "item_id" not in products_df.columns:
        raise ValueError("Products DataFrame must contain item_id.")

    if products_df["item_id"].isna().any():
        raise ValueError("Product data contains missing item IDs.")

    if products_df["item_id"].duplicated().any():
        raise ValueError("Product data contains duplicate item IDs.")

    # 3. Build embedding texts
    print("Building embedding texts...")

    embedding_text = build_text_embedding(products_df)

    if len(embedding_text) != len(products_df):
        raise ValueError(
            "Number of embedding texts does not match number of products."
        )

    print(f"Created {len(embedding_text)} embedding texts.")

    # 4. Load embedding model
    print(f"Loading embedding model: {embedding_model_name}")

    embedding_service = EmbeddingService(
        embedding_model_name
    )

    print(f"Embedding dimension: {embedding_service.dimension}")

    # 5. Generate product embeddings
    print("Generating embeddings...")

    embeddings = embedding_service.encode(
        embedding_text,
        batch_size=32
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # 6. Prepare product ID mapping
    # Each ID must correspond to the embedding at the same position.
    item_ids = products_df["item_id"].astype(str).to_numpy(dtype=np.str_)

    # 7. Validate generated artifacts
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if embeddings.shape[0] != len(item_ids):
        raise ValueError(
            "Number of embeddings does not match number of product IDs."
        )

    if embeddings.shape[1] != embedding_service.dimension:
        raise ValueError(
            "Embedding dimension does not match the model dimension."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    # 8. Create models directory
    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 9. Save product ID mapping
    item_ids_file = models_dir / "item_ids.npy"

    np.save(
        item_ids_file,
        item_ids
    )

    # 10. Save product embeddings
    np.save(
        item_embeddings_file,
        embeddings
    )

    # 11. Display artifact information
    print("\nEmbedding generation completed successfully.")

    print(f"Products: {len(products_df)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Product ID shape: {item_ids.shape}")
    print(f"Unique product IDs: {len(np.unique(item_ids))}")

    print(f"\nEmbeddings saved to: {item_embeddings_file}")
    print(f"Product IDs saved to: {item_ids_file}")


if __name__ == "__main__":
    main()
