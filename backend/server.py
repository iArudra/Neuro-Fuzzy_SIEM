import sys
import os
import json
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List

from database import engine, SessionLocal, init_db, DBLogEntry, DBAlert
from predictor import StatefulPredictor

app = FastAPI(title="SIEM Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = StatefulPredictor()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Broadcast to all connected clients
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        for d in disconnected:
            self.disconnect(d)

manager = ConnectionManager()

class LogIngestRequest(BaseModel):
    logs: List[str]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/ingest")
async def ingest_logs(request: LogIngestRequest, db: Session = Depends(get_db)):
    results = []
    for log_str in request.logs:
        pred = predictor.ingest_log(log_str)
        if not pred:
            continue
            
        parsed = pred["parsed"]
        # Save log entry
        db_log = DBLogEntry(
            timestamp=parsed["datetime"],
            raw_text=log_str,
            level="ERROR" if parsed["is_error"] else "INFO",
            component=parsed["block_id"]
        )
        db.add(db_log)
        
        score = pred["risk_score"]
        level = pred["alert_level"]
        
        db_alert = None
        # Only log alerts for Medium+ severity
        if level in ["Medium", "High", "Critical"]:
            db_alert = DBAlert(
                timestamp=parsed["datetime"],
                risk_score=score,
                alert_level=level,
                triggering_logs=json.dumps(pred["window_logs"])
            )
            db.add(db_alert)
            
        db.commit()
        db.refresh(db_log)
        if db_alert:
            db.refresh(db_alert)
            
        res = {
            "type": "update",
            "log": {
                "id": db_log.id,
                "timestamp": db_log.timestamp.isoformat(),
                "raw_text": db_log.raw_text,
                "level": db_log.level,
                "component": db_log.component
            },
            "risk_score": score,
            "alert": None
        }
        
        if db_alert:
            res["alert"] = {
                "id": db_alert.id,
                "timestamp": db_alert.timestamp.isoformat(),
                "risk_score": db_alert.risk_score,
                "alert_level": db_alert.alert_level,
            }
            
        results.append(res)
        await manager.broadcast(res)
        
    return {"status": "ok", "processed": len(results)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Just keep the connection open, we don't expect messages from client
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total_logs = db.query(DBLogEntry).count()
    active_alerts = db.query(DBAlert).count()
    latest_alert = db.query(DBAlert).order_by(DBAlert.timestamp.desc()).first()
    return {
        "total_logs": total_logs, 
        "active_alerts": active_alerts,
        "latest_alert_level": latest_alert.alert_level if latest_alert else "None"
    }

@app.get("/recent_alerts")
def get_recent_alerts(limit: int = 50, db: Session = Depends(get_db)):
    alerts = db.query(DBAlert).order_by(DBAlert.timestamp.desc()).limit(limit).all()
    return alerts

@app.get("/recent_logs")
def get_recent_logs(limit: int = 100, db: Session = Depends(get_db)):
    logs = db.query(DBLogEntry).order_by(DBLogEntry.timestamp.desc()).limit(limit).all()
    return logs
