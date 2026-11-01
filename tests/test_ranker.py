
import pandas as pd
import pytest

from app.ranker import Ranker


@pytest.fixture
def candidates():
    """Sample products for testing the ranker."""
    return pd.DataFrame({
        "item_id": [1, 2, 3, 4],
        "similarity_score": [0.95, 0.90, 0.85, 0.80],
        "average_rating": [4.8, 4.5, 4.2, 4.7],
        "total_reviews": [1000, 50, 20, 80],
        "total_sales": [500, 250, 100, 400],
        "created_at": [
            "2026-10-10",
            "2026-09-20",
            "2026-08-15",
            "2026-10-01",
        ],
        "category": [
            "electronics",
            "electronics",
            "fashion",
            "books",
        ],
    })


@pytest.fixture
def ranker():
    """Create a ranker with default settings."""
    return Ranker(top_k=3, max_per_category=2)


def test_validate_columns(ranker, candidates):
    ranker.validate_candidates(candidates)

    invalid = candidates.drop(columns=["similarity_score"])

    with pytest.raises(ValueError):
        ranker.validate_candidates(invalid)


def test_prepare_features(ranker, candidates):
    result = ranker.prepare_feature(candidates)

    assert "similarity_signal" in result.columns
    assert "rating_signal" in result.columns
    assert "freshness_signal" in result.columns
    assert "popularity_signal" in result.columns

    assert result["similarity_signal"].between(0, 1).all()
    assert result["rating_signal"].between(0, 1).all()
    assert result["freshness_signal"].between(0, 1).all()
    assert result["popularity_signal"].between(0, 1).all()


def test_calculate_relevance_score(ranker, candidates):
    result = ranker.prepare_feature(candidates)
    result = ranker.calculate_relevance_score(result)

    assert "base_score" in result.columns
    assert result["base_score"].notna().all()
    assert result["base_score"].between(0, 1).all()


def test_similarity_has_highest_weight(ranker):
    assert ranker.weights["similarity"] > ranker.weights["rating"]
    assert ranker.weights["similarity"] > ranker.weights["popularity"]
    assert ranker.weights["similarity"] > ranker.weights["fresheness"]


def test_category_diversity(ranker, candidates):
    result = ranker.prepare_feature(candidates)
    result = ranker.calculate_relevance_score(result)
    result = ranker.apply_category_diversity(result, top_k=3)

    assert len(result) <= ranker.top_k
    assert result["category"].value_counts().max() <= 2


def test_ranker_returns_top_k(ranker, candidates):
    result = ranker.rank(candidates)

    assert len(result) == 3
    assert result["item_id"].is_unique
    assert result["rank"].tolist() == [1, 2, 3]
    assert "base_score" in result.columns


def test_rank_orders_products_by_score(ranker, candidates):
    result = ranker.rank(candidates)

    assert result["base_score"].is_monotonic_decreasing


def test_empty_candidates(ranker):
    empty = pd.DataFrame(columns=["item_id", "similarity_score"])

    result = ranker.rank(empty)

    assert result.empty


def test_invalid_similarity_score(ranker, candidates):
    candidates.loc[0, "similarity_score"] = float("inf")

    with pytest.raises(ValueError):
        ranker.rank(candidates)


def test_input_is_not_modified(ranker, candidates):
    original = candidates.copy(deep=True)

    ranker.rank(candidates)

    pd.testing.assert_frame_equal(candidates, original)
