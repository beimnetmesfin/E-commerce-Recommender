import pytest
import numpy as np
import pandas as pd
from app.retrieval import Retriever
@pytest.fixture
def products():
    return pd.DataFrame(
        {
            "item_id":["yb001","yb002","yb003","yb004"],
            "title":["coffee","laptop bag","coffee beans","phone case"],
            "category":["food","fashion","food","electonics"]
        }
    )

class FakeFaissService:
    def search(
            self,query_embedding,k=10
    ):
        scores=np.array([[0.95,0.90,0.80]],dtype=np.float32)
        indices=np.array([[2,0,3]],dtype=int)
        return scores,indices
@pytest.fixture
def retriever():
    faiss_service=FakeFaissService()
    return Retriever(faiss_service)

@pytest.fixture
def user_profile():
    return np.array([0.8,0.2,0.1],dtype=np.float32)

def test_retrieve_returns_candidates(
       retriever, products,user_profile
):
    candidates=retriever.retrieve(user_profile,products,k=3)
    assert isinstance(candidates,pd.DataFrame)
    assert len(candidates)==3
def test_retrieve_returns_correct_products(
        retriever,products,user_profile
):
    candidates=retriever.retrieve(user_profile,products,k=3)
    assert candidates["item_id"].tolist()==["yb003","yb001","yb004"]
def test_retrieve_attaches_similarity_scores(
        retriever,products,user_profile
):
    candidates=retriever.retrieve(user_profile,products,k=3)
    expected=[0.95,0.90,0.80]
    assert np.allclose(candidates["similarities"].values,expected)
def test_retrieve_resets_index(
        retriever,products,user_profile
):
    candidates=retriever.retrieve(user_profile,products,k=3)
    assert candidates.index.to_list()==[0,1,2]
def test_empty_products_raise_error(retriever,products,user_profile):
    empyt_products=pd.DataFrame()
    with pytest.raises(ValueError):
        retriever.retrieve(user_profile,empyt_products,k=3)
def test_invalid_profile_dimension(retriever,products):
    invalid_profile=np.array([[0.8,0.2,0.1]],dtype=np.float32)
    with pytest.raises(ValueError):
        retriever.retrieve(invalid_profile,products,k=3)
