
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.event_database import (
    create_event_table,
    save_event,
    get_events,
)

from app.recommendation_engine import (
    recommend_for_user,
    get_similar_products,
)


app = FastAPI(
    title="Ye-buna Recommendation API",
    description="Recommendation API with user behavior tracking",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EventInput(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    item_id: int = Field(gt=0)
    event_type: Literal[
        "view",
        "click",
        "like",
        "add_to_cart",
        "purchase",
        "skip",
    ]


@app.on_event("startup")
def initialize_database():
    create_event_table()


@app.get("/")
def home():
    return {
        "message": "Ye-buna Recommendation API is running",
        "status": "ok",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/events")
def create_event(event: EventInput):

    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        save_event(
            user_id=event.user_id.strip(),
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

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not save event",
        ) from exc


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
            "recommendations": recommendations.to_dict(
                orient="records"
            ),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.get("/items/{item_id}/similar")
def get_similar_items(
    item_id: int,
    top_k: int = Query(default=4, ge=1, le=50),
):
    try:
        similar_products = get_similar_products(
            item_id=item_id,
            top_k=top_k,
        )

        return {
            "item_id": item_id,
            "count": len(similar_products),
            "similar_products": similar_products.to_dict(
                orient="records"
            ),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
