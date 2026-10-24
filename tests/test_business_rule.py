import pandas as pd
from app.business_rule import BusinessRules

def test_filter_active():
    candidates=pd.DataFrame(
        {
            "item_id":[1,2],
            "is_active":[True,False]
        }
    )
    rules=BusinessRules()
    result=rules.filter_active(candidates)
    assert result["item_id"].tolist()==[1]

def test_filter_visible():
    candidates=pd.DataFrame(
        {
            "item_id":[1,2],
            "is_visible":[True,False]
        }
    )
    rules=BusinessRules()
    result=rules.filter_visible(candidates)
    assert result["item_id"].tolist()==[1]

def test_filter_approved():
    candidates=pd.DataFrame(
        {
            "item_id":[1,2],
            "approval_status":["approved","pending"]
        }
    )
    rules=BusinessRules()
    result=rules.filter_approved(candidates)
    assert result["item_id"].tolist()==[1]

def test_filter_stock_in():
    candidates=pd.DataFrame(
        {
            "item_id":[1,2,3],
            "stock":[10,0,8],
            "reserved_stock":[2,0,4],
            "inventory_status":["available","out_of_stock","discontinued"]
        }
    )
    rules=BusinessRules()
    result=rules.filter_in_stock(candidates)
    assert result["item_id"].tolist()==[1,3]

def test_filter_not_purchased():
    candidates=pd.DataFrame(
            {
                "item_id":[1,2,3,4],
                "stock":[10,0,8,5],
                "reserved_stock":[2,0,4,0],
                "inventory_status":["available","out_of_stock","discontinued","available"]
            })
    user_events=pd.DataFrame({
        "user_id":[1,2,3,4],
        "item_id":[1,2,3,4],
        "event_type":["purchase","like","like","like"],
        "event_status":["completed","delivered","paid","pending"]
    })

    rules=BusinessRules()
    result=rules.filter_not_purchased(candidates,user_events["user_id"],user_events)
    assert result["item_id"].to_list()==[2,3,4]
    