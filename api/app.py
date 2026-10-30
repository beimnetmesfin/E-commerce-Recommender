
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.event_database import (
    create_event_table,
    save_event,
    get_events,
)
from app.recommendation_engine import recommend_for_user,get_similar_products


app = FastAPI(
    title="E-commerce Recommendation API",
    description="API for the E-commerce product recommendation platform",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EventInput(BaseModel):
    user_id: str
    item_id: str
    event_type: str


@app.on_event("startup")
def initialize_database():
    create_event_table()


@app.get("/")
def home():
    return {
        "message": "E-commerce Recommendation API is running",
        "status": "ok",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/events")
def create_event(event: EventInput):
    timestamp = datetime.now(timezone.utc).isoformat()

    save_event(
        user_id=event.user_id,
        item_id=event.item_id,
        event_type=event.event_type,
        timestamp=timestamp,
    )

    return {
        "status": "saved",
        "user_id": event.user_id,
        "item_id": event.item_id,
        "event_type": event.event_type,
        "timestamp": timestamp,
    }


@app.get("/events")
def list_events():
    return get_events()


@app.get("/recommendations/{user_id}")
def get_recommendations(
    user_id: str,
    top_k: int = Query(default=10, ge=1, le=50),
):
    try:
        recommendations = recommend_for_user(
            user_id=user_id,
            top_k=top_k,
        )

        return {
            "user_id": user_id,
            "count": len(recommendations),
            "recommendations": recommendations.to_dict(orient="records"),
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/items/{item_id}/similar")
def get_similar_items(
    item_id:int,
    top_k:int=Query(default=4,ge=1,le=50),
):
    try:
        similar_products=get_similar_products(
            item_id=item_id,top_k=top_k
        )
        return {
            "item_id":item_id,
            "count":len(similar_products),
            "similar_products":similar_products.to_dict(
                orient="records"
            )
        }
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
