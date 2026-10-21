from typing import Optional
import pandas as pd
import numpy as np

class Ranker:
    """
    Ranker egilibale product candidates and select a diverse Top-k list
    Resposnebilities
    1 Prepare the products feature of validates ranking inputs
    2 prepare the normalized ranking features
    3 calculate a weighted relevance score
    4 sort the candidates by relevalce
    5 Apply category diversity
    6 Return the final recommendations
    """
    reqiured_columns=["item_id","similarities"]
    default_weights={
        "similarity":0.70,
        "rating":0.10,
        "popularity":0.10,
        "fresheness":0.10
    }
    def __init__(self,
                 top_k:int=10,
                 max_per_category:Optional[int]=3,
                 weights:Optional[dict]=None,
                 rating_prior:float=3.5,
                 prior_reviews:int=10,
                 freshness_haf_life_days:int=180)->None:
        if top_k<1:
            raise ValueError("top_k must be at least 1.")
        if(max_per_category is not None and max_per_category<1):
            raise ValueError("max_per_category must be at least 1 or None")
        if prior_reviews<0:
            raise ValueError("prior_reviews cannot be negative.")
        if freshness_haf_life_days<1:
            raise ValueError("freshness_haf_life_days must be at least 1.")

        self.top_k=top_k
        self.max_per_category=max_per_category
        self.rating_prior=rating_prior
        self.prior_reviews=prior_reviews
        self.freshness_haf_life_days=(freshness_haf_life_days)
        self.weights=(weights.copy()
                              if weights is not None
                              else self.default_weights.copy()
                              )
        if any(
            not np.isfinite(weight) or weight <0
            for weight in self.weights.values()):
            raise ValueError("weights must be finite, non-negative numbers")
        if sum(self.weights.values())<=0:
            raise ValueError("At least one weight must be positive.")
    def validate_candidates(self,candidates:pd.DataFrame)->None:
            if not isinstance(candidates,pd.DataFrame):
                raise ValueError("cadidates must be a pandas DataFrame")
            
            missing_columns=[col for col in self.reqiured_columns 
                             if col not in candidates.columns]
            if missing_columns:
                raise ValueError(f"Missing required ranking columns :{missing_columns}")
            if candidates["item_id"].isna().any():
                raise ValueError("Item_id cannot contain missing values")
            if candidates["item_id"].duplicated().any():
                raise ValueError("The candidates must contain unique item_id and cannot duplicate ")

    def prepare_feature(self,candidates:pd.DataFrame)->pd.DataFrame:  
            """creating normalized scoring feature"""
            ranked=candidates.copy()
            #FAISS Similarity between 0.0 and 1.0 for scoring
            similarity=pd.to_numeric(ranked["similarities"],errors="coerce")
            similarity=similarity.replace(
                [np.inf,-np.inf],np.nan
            )
            #similarity=similarity.fillna(0.0)
            if similarity.isna().any():
                raise ValueError("similarty_score must contain valid finite values")
            ranked["similarity_signal"]=similarity.clip(
                lower=0.0,upper=1.0)
            #Rating quality :account for both the average rating and return of reviews
            if "average_rating" in ranked.columns:
                rating=pd.to_numeric(
                    ranked["average_rating"],
                    errors="coerce"
                )
                if(rating.notna()& ~np.isfinite(rating)).any():
                    raise ValueError("average_rating contains infinite values")
                    rating=rating.where(rating.between(0,5),np.nan)
                if "total_reviews" in ranked.columns:
                    review_counts=pd.to_numeric(ranked["total_reviews"],
                                    errors="coerce")
                    review_counts=review_counts.where(
                        np.isfinite(review_counts)&(review_counts>=0),
                         np.nan
                         ).fillna(0)
                else:
                    review_counts=pd.Series(0.0,index=ranked.index)
                prior=self.prior_reviews
                total_weight=review_counts+prior
                if prior>0:
                    rating_score=((review_counts*rating)+(prior*self.rating_prior))/total_weight.replace(0,np.nan)
                else:                
                   rating_score=rating
                ranked["rating_signal"]=(rating_score/5.0).fillna(0.5).clip(0.0,1.0)
            #pupularity :log scaling prevents products with
            #extremely high sales from dominating the ranking
            if "total_sales" in ranked.columns:
                sales=pd.to_numeric(ranked["total_sales"],errors="coerce")
                sales=sales.where(np.isfinite(sales)&(sales>=0),np.nan)
                max_sales=sales.max()
                if pd.isna(max_sales) or max_sales<=0:
                    popularity_score=pd.Series(0.0,index=ranked.index)
                else:
                    popularity_score=(np.log1p(sales.fillna(0))/np.log1p(max_sales))
                 
                ranked["popularity_signal"]=(popularity_score.clip(0.0,1.0))

            #Freshness :New products recieve a higher signal
            #The score halves every configured half-life peroid

            if "created_at" in ranked.columns:
                created_at=pd.to_datetime(ranked["created_at"],errors="coerce",utc=True)
                now=pd.Timestamp.now(tz="UTC")
                age_days=((now-created_at).dt.total_seconds()/86400.0)
                age_days=age_days.where(age_days>=0,np.nan)
                freshness_score=(0.5**(age_days/self.freshness_haf_life_days))
                ranked["freshness_signal"]=(freshness_score.fillna(0.5).clip(0.0,1.0))
            return ranked

    def calculate_relevance_score(self,candidates:pd.DataFrame)->pd.DataFrame:
        ranked=candidates.copy()
        signal_columns={
               "similarity":"similarity_signal",
               "rating":"rating_signal",
               "popularity":"popularity_signal",
               "freshness":"freshness_signal"
           }
        #only use the features that were acutually created
        active_weights={
        name:weight
        for name,weight in self.weights.items()
             if(
                   name in signal_columns and
                   signal_columns[name] in ranked.columns and
                   weight>0
               )
           }
        if not active_weights:
            raise ValueError("No usable ranking signals are available ")
        total_weight=sum(active_weights.values())
        #Renormalized the total weights

        normalized_weight={
           name:weight/total_weight
            for name,weight in active_weights.items()
           }
        ranked["base_score"]=0.0
        for name,weight in normalized_weight.items():
            column=signal_columns[name]
            ranked["base_score"]+=(ranked[column]*weight)

        return ranked
    def apply_category_diversity(self,ranked:pd.DataFrame,top_k:int)->pd.DataFrame:
        #Selecting high scoring in limited category
        ordered=ranked.sort_values(
            by=["base_score","similarity_signal","item_id"],
            ascending=[False,False,True],
            kind="mergesort"
        )
        if self.max_per_category is None:
            return ordered.head(top_k).copy()
        if "category" not in ranked.columns:
            return ordered.head(top_k).copy()

        selected_indices=[]
        skipped_indices=[]
        category_counts={}
        #phase 1 enforch the category limit
        for index,row in ordered.iterrows():
            if len(selected_indices)>=top_k:
                break
            category=row["category"]
            if pd.isna(category) or not str(category).strip():
                #unknown categories do not all share one limit
                category=f"__unknown_{row["category"]}"
            count=category_counts.get(category,0)
            if count<self.max_per_category:
                selected_indices.append(index)
                category_counts[category]=count+1
            else:
                skipped_indices.append(index)
        #pass 2 if the limit prevented us from filling Top-k
        #fill remaining slots with the best skipped candidates
        remaining=top_k-len(selected_indices)
        if remaining>0:
            selected_indices.extend(skipped_indices[:remaining])
        return ordered.loc[selected_indices].copy()
    def rank(self,candidates:pd.DataFrame,top_k:Optional[int]=None
               )->pd.DataFrame:
        """Run the complete ranking pipeline"""
        self.validate_candidates(candidates)
        requested_k=(self.top_k if top_k is None else top_k)
        if requested_k<1:
            raise ValueError("top_k must be at least 1")
        if candidates.empty:
            return candidates.copy()
        ranked=self.prepare_feature(candidates)
        ranked=self.calculate_relevance_score(ranked)
        ranked=self.apply_category_diversity(ranked,top_k=requested_k)
        ranked=ranked.reset_index(drop=True)
        ranked["rank"]=np.arange(1,len(ranked)+1)
        return ranked

