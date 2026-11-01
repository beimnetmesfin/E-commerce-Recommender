
import numpy as np
import pandas as pd
import pytest

from app.retrieval import Retriever


@pytest.fixture
def products():
    return pd.DataFrame({
        "item_id": ["yb001", "yb002", "yb003", "yb004"],
        "title": ["coffee", "laptop bag", "coffee beans", "phone case"],
        "category": ["food", "fashion", "food", "electronics"],
    })


class FakeIndex:
    """Minimal index metadata used by Retriever validation."""

    ntotal = 4
    d = 3


class FakeFaissService:
    def __init__(self):
        self.index = FakeIndex()

    def search(self, query_embedding, k=10):
        scores = np.array(
            [[0.95, 0.90, 0.80]],
            dtype=np.float32,
        )
        indices = np.array(
            [[2, 0, 3]],
            dtype=np.int64,
        )
        return scores, indices


@pytest.fixture
def retriever(tmp_path):
    item_ids_file = tmp_path / "item_ids.npy"

    np.save(
        item_ids_file,
        np.array(
            ["yb001", "yb002", "yb003", "yb004"],
            dtype=np.str_,
        ),
    )

    return Retriever(
        FakeFaissService(),
        item_ids_file,
    )


@pytest.fixture
def user_profile():
    return np.array([0.8, 0.2, 0.1], dtype=np.float32)


def test_retrieve_returns_candidates(retriever, products, user_profile):
    candidates = retriever.retrieve(
        user_profile,
        products,
        k=3,
    )

    assert isinstance(candidates, pd.DataFrame)
    assert len(candidates) == 3


def test_retrieve_returns_correct_products(
    retriever,
    products,
    user_profile,
):
    candidates = retriever.retrieve(
        user_profile,
        products,
        k=3,
    )

    assert candidates["item_id"].tolist() == [
        "yb003",
        "yb001",
        "yb004",
    ]


def test_retrieve_attaches_similarity_scores(
    retriever,
    products,
    user_profile,
):
    candidates = retriever.retrieve(
        user_profile,
        products,
        k=3,
    )

    expected = [0.95, 0.90, 0.80]

    assert np.allclose(
        candidates["similarity_score"].values,
        expected,
    )


def test_retrieve_resets_index(retriever, products, user_profile):
    candidates = retriever.retrieve(
        user_profile,
        products,
        k=3,
    )

    assert candidates.index.to_list() == [0, 1, 2]


def test_empty_products_raise_error(retriever, user_profile):
    empty_products = pd.DataFrame()

    with pytest.raises(ValueError):
        retriever.retrieve(
            user_profile,
            empty_products,
            k=3,
        )


def test_invalid_profile_dimension(retriever, products):
    invalid_profile = np.array(
        [[0.8, 0.2, 0.1]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            invalid_profile,
            products,
            k=3,
        )
