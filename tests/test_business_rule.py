import pandas as pd
import pytest

from app.business_rule import BusinessRules


def test_filter_in_stock():
    candidates = pd.DataFrame({
        "item_id": [1, 2, 3],
        "stock": [10, 0, 5],
    })

    rules = BusinessRules()
    result = rules.filter_in_stock(candidates)

    assert result["item_id"].tolist() == [1, 3]


def test_remove_duplicates():
    candidates = pd.DataFrame({
        "item_id": [1, 2, 1, 3],
        "stock": [10, 5, 10, 8],
    })

    rules = BusinessRules()
    result = rules.remove_duplicates(candidates)

    assert result["item_id"].tolist() == [1, 2, 3]


def test_validate_products_columns():
    candidates = pd.DataFrame({
        "item_id": [1, 2],
        "stock": [10, 5],
    })

    rules = BusinessRules()

    rules.validate_products_columns(candidates)


def test_validate_missing_products_columns():
    candidates = pd.DataFrame({
        "item_id": [1, 2],
    })

    rules = BusinessRules()

    with pytest.raises(ValueError):
        rules.validate_products_columns(candidates)


def test_apply_rules():
    candidates = pd.DataFrame({
        "item_id": [1, 2, 1, 3, 4],
        "stock": [10, 0, 10, 5, -1],
    })

    rules = BusinessRules()

    result = rules.apply_rules(
        candidates=candidates,
        user_id="user_001",
        user_events=pd.DataFrame(),
    )

    assert result["item_id"].tolist() == [1, 3]
    assert result["stock"].tolist() == [10, 5]
    assert result.index.tolist() == [0, 1]