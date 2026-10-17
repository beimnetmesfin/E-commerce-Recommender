import pandas as pd

class Recommender:
    """Connect all existing componets into one recommendation pipeline"""
    def __init__(self,load_products,load_events,
                 build_profile,retrieve_candidates,
                 apply_business_rule,ranker
                 ):
        self.load_products=load_products
        self.load_events=load_events
        self.build_profile=build_profile
        self.retrieve_candidates=retrieve_candidates
        self.apply_business_rule=apply_business_rule
        self.ranker=ranker
    def recommend_for_user(self,user_id,top_k):
        if not isinstance(user_id,str) or not user_id.strip():
            raise ValueError("user id must be a non-empty string")

        products=self.load_products()
        events=self.load_events()
        user_events=events[events["user_id"]==user_id]

        if user_events.empty:
            return pd.DataFrame()
        profile=self.build_profile(user_events)
        
        if profile is None:
            return pd.DataFrame()
        candidates=self.retrieve_candidates(profile,products)
        if candidates is None or candidates.empty:
            return pd.DataFrame()
        eligible=self.apply_business_rule(candidates,products)
        if eligible is None or eligible.empty:
            return pd.DataFrame()
        if top_k is None:
            return self.ranker.rank(eligible)
        return self.ranker.rank(eligible,top_k=top_k)