from pathlib import Path

#Project Root
base_dir=Path(__file__).resolve().parent.parent

#Create the folders of the datasets

data_dir=base_dir/"data"
models_dir=base_dir/"models"
test_index=base_dir/"test_folder"/"test.index"
#Create the files in the data folder of the above
products_file=data_dir/"products.csv"
user_events_file=data_dir/"user_events.csv"

#Create the files in the models folder of the above

item_embeddings_file=models_dir/"item_embeddings.npy"
faiss_index_file=models_dir/"faiss_index.index"
item_ids_file = models_dir / "item_ids.npy"

embedding_model_name=("sentence-transformers/"
"paraphrase-multilingual-MiniLM-L12-v2")

embedding_dimension=384
retrival_number=50
top_recommendation=10

