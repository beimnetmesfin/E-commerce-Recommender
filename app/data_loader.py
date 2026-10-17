import pandas as pd
from pathlib import Path
#from app.config import products_file,user_events_file
class DataLoader:
    def __init__(self,products_path:Path,user_events_path:Path):
        self.products_path=Path(products_path)
        self.user_events_path=Path(user_events_path)

    def _read_csv(self,path:Path)-> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"File is not found in this path {path}")
        return pd.read_csv(path,sep="\t")
    
    def validate_columns(self,df:pd.DataFrame,required_columns:list,name:str)->None:
        missing=[col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"{name} is the missing required columns {missing}")
        
    def load_products(self,clean:bool=True)->pd.DataFrame:
        df=self._read_csv(self.products_path)

        required_columns = [
            "item_id",
            "title",
            "description",
            "category",
            "subcategory",
            "tags",
            "creator_name",
            "creator_rating",
            "price",
            "currency",
            "language",
            "country",
            "stock",
            "created_at",
            "average_rating",
            "total_reviews",
            "total_sales"
            "image_url"
        ]

        if clean:
            df=df.copy()
            #text standaridze and data cleaning
            text_columns=[
                "title",
                "description",
                "category",
                "subcategory",
                "tags",
                "creator_name",
                "currency",
                "language",
                "image_url",
                "country"
            ]
            for col in text_columns:
                df[col]=df[col].fillna("").astype(str).str.strip()

            #numeric clean
            numeric_columns=[
                "item_id",
                "price",
                "stock",
                "average_rating",
                "total_reviews",
                "total_sales",
                "creator_rating"
            ]
            for col in numeric_columns:
                df[col]=pd.to_numeric(df[col],errors="coerce")
            # to staye the data types for the numberic columns
            numeric_float=["average_rating","creator_rating","price"]
            numeric_int=["stock","total_reviews","total_sales"]
            for col in numeric_float:
                df[col]=df[col].fillna(0.0)
            for col in numeric_int:
                df[col]=df[col].fillna(0).astype(int)

            #parse the datetime 
            df["created_at"]=pd.to_datetime(df["created_at"],errors="coerce")

            #Remove the duplicates in the dataset
            df=df.drop_duplicates(subset=["item_id"],inplace=False).reset_index(drop=True)

        return df

    def load_user_events(self,clean:bool=True)->pd.DataFrame:
        df=self._read_csv(self.user_events_path)

        required_columns=[
            "user_id",
            "item_id",
            "event_type",
            "recency_rank"
        ]
        self.validate_columns(df,required_columns,"user_events.csv")

        if clean:
            df=df.copy()
            numeric_col=[
                        "user_id",
                        "item_id",
                        "recency_rank"
                    ]
            for col in numeric_col:
                df[col]=pd.to_numeric(df[col],errors="coerce")

            df["event_type"]=df["event_type"].fillna("").astype(str).str.strip()
            
            df=df.dropna(subset=["user_id","item_id","event_type"])
            df["item_id"]=df["item_id"].astype(int)
            df["user_id"]=df["user_id"].astype(int)
            df["recency_rank"]=df["recency_rank"].fillna(0).astype(int)

            df=df.drop_duplicates().reset_index(drop=True)
        return df

