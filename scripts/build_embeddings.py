import numpy as np
from pathlib import Path
from app.config import (products_file,user_events_file,
                        embedding_model_name,models_dir,item_embeddings_file)

from app.data_loader import DataLoader
from app.text_builder import build_text_embedding
from app.embedding_service import EmbeddingService

def main()->None:

    print("Loading the product data")
    loader=DataLoader(products_file,user_events_file)

    products_df=loader.load_products()
    print(f"Loaded {len(products_df)} products.")

    print("Building the embedding texts")
    embedding_text=build_text_embedding(products_df)
    print(f"Created {len(embedding_text)} embedding texts")

    print(f"Loading embedding model: {embedding_model_name}")
    embedding_service=EmbeddingService(embedding_model_name)

    print(f"Embedding Dimension :{embedding_service.dimension}")

    #Generating embedding texts
    print(f"Generating embedding")

    embeddings=embedding_service.encode(embedding_text,batch_size=32)

    #Get information about the embeddings
    print(f"embedding shape:{embeddings.shape}")
    print(f"embedding dtype:{embeddings.dtype}")
    print(embeddings)

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    np.save(
        item_embeddings_file,
        embeddings
    )

    print(f"embedding saved to:{item_embeddings_file}")


if __name__=="__main__":
    main()
