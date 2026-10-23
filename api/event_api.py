
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Ye-buna Event API")

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

@app.post("/events")
def receive_event(event: EventInput):
    return {
        "status": "received",
        "user_id": event.user_id,
        "item_id": event.item_id,
        "event_type": event.event_type,
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
