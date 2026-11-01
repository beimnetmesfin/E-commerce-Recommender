
import pandas as pd


class BusinessRules:

    required_products_columns = [
        "item_id",
        "stock",
    ]

    def validate_products_columns(
        self,
        candidates: pd.DataFrame,
    ) -> None:

        missing_columns = [
            col
            for col in self.required_products_columns
            if col not in candidates.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required product columns: {missing_columns}"
            )

    def filter_in_stock(
        self,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:

        stock = pd.to_numeric(
            candidates["stock"],
            errors="coerce",
        )

        return candidates[stock > 0].copy()

    def remove_duplicates(
        self,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:

        return candidates.drop_duplicates(
            subset=["item_id"]
        ).copy()

    def apply_rules(
        self,
        candidates: pd.DataFrame,
        user_id,
        user_events: pd.DataFrame,
    ) -> pd.DataFrame:

        self.validate_products_columns(candidates)

        candidates = self.filter_in_stock(candidates)

        candidates = self.remove_duplicates(candidates)

        return candidates.reset_index(drop=True)
