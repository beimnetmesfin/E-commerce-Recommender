import pandas as pd
semantic_columns=[
    "title",
    "description",
    "category",
    "subcategory",
    "tags",
    "language",
    "country",
    "creator_name",
]

def validate_semantic_columns(df:pd.DataFrame)->None:
    missing_columns=[col for col in semantic_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"missing semantic columns :{missing_columns}")

def create_text_embedding(row:pd.Series)->str:
    #creates semantic text representation for one product
    return (
        f"Title:{row["title"]}."
        f"Description:{row["description"]}."
        f"Category:{row["category"]}."
        f"Subcatogory:{row["subcategory"]}."
        f"Tags:{row["tags"]}"
        f"Creator:{row["creator_name"]}"
        f"Language:{row["language"]}"
        f"Country:{row["country"]}"
    )

def build_text_embedding(df:pd.DataFrame)->list[str]:
    #Build embedding text for every product
    validate_semantic_columns(df)
    return [create_text_embedding(row) for _,row in df.iterrows()]
