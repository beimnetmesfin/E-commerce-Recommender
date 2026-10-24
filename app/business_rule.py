import pandas as pd
class BusinessRules:
    completed_order_status=["completed","delivered","paid","confirmed"]
    required_events_columns=["user_id","item_id","event_type","event_status"]
    required_products_columns=["item_id","stock"]
    def validate_products_columns(self,candidates:pd.DataFrame)->None:
        missing_columns=[col for col in self.required_products_columns if col not in candidates.columns]
        if missing_columns:
            raise ValueError(f"missing required columns :{missing_columns}")
    def validate_events_columns(self,user_events:pd.DataFrame)->None:
        missing_columns=[col for col in self.required_events_columns if col not in user_events.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns :{missing_columns}")

    def filter_active(self,candidates:pd.DataFrame)->pd.DataFrame:
        return candidates[candidates["is_active"]==True].copy()
    def filter_visible(self,candidates:pd.DataFrame)->pd.DataFrame:
        return candidates[candidates["is_visible"]==True].copy()
    def filter_approved(self,candidates:pd.DataFrame)->pd.DataFrame:
        return candidates[candidates["approval_status"]=="approved"].copy()
    def filter_not_deleted(self,candidates:pd.DataFrame)->pd.DataFrame:
        return candidates[candidates["deleted_at"].isna()].copy()

    def filter_available_now(self,candidates:pd.DataFrame)->pd.DataFrame:
        now=pd.Timestamp.now()
        available_from=pd.to_datetime(candidates["available_from"],errors="coerce")
        available_until=pd.to_datetime(candidates["available_until"],errors="coerce")
        valid_from=(available_from.isna()|(available_from<=now))
        valid_until=(available_until.isna()|(available_until>=now))
        return candidates[valid_from &valid_until].copy()

    def filter_in_stock(self,candidates:pd.DataFrame)->pd.DataFrame:
        #available_stock=(candidates["stock"]-candidates["reserved_stock"])
        #The inventory status should be in the website code or in kafka
        #valid_status=~candidates["inventory_status"].isin(
         # ["out_of_stock","unavailable","discontinued"])
        available_stock=pd.to_numeric(candidates["stock"],errors="coerce")
        return candidates[(available_stock>0)].copy()
    
    def filter_not_purchased(self,candidates:pd.DataFrame,
                             user_id,
                             user_events:pd.DataFrame)->pd.DataFrame:
        successful_purchase=user_events[
            (user_events["user_id"]==user_id)&
            (user_events["event_type"]=="purchase")&
            (user_events["event_status"].isin(self.completed_order_status))
        ]
        purchased_items_id=set(successful_purchase["item_id"])
        return candidates[~candidates["item_id"].isin(purchased_items_id)].copy()

    def remove_duplicates(self,candidates:pd.DataFrame)->pd.DataFrame:
        return candidates.drop_duplicates(subset=["item_id"]).copy()
    
    def apply_rules(self,candidates:pd.DataFrame,
                    user_id,
                    user_events:pd.DataFrame)->pd.DataFrame:

        self.validate_products_columns(candidates)
        #self.validate_events_columns(user_events)
        #candidates=self.filter_active(candidates)
        #candidates=self.filter_visible(candidates)
        #candidates=self.filter_approved(candidates)
        #candidates=self.filter_not_deleted(candidates)
        #candidates=self.filter_available_now(candidates)
        #candidates=self.filter_not_purchased(candidates,user_id,user_events)
        candidates=self.filter_in_stock(candidates)
        candidates=self.remove_duplicates(candidates)
        return candidates
