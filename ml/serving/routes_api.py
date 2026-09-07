from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
from ml.serving.auth import get_current_user
from ml.serving.database import get_db

router = APIRouter(prefix="/api", tags=["api"])

class AlertUpdate(BaseModel):
    status: str

@router.get("/forecasts")
async def get_forecasts(current_user: dict = Depends(get_current_user), db = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    forecasts = await db["forecasts"].find({"user_id": current_user["id"]}).sort("timestamp", -1).to_list(100)
    return forecasts

@router.get("/forecasts/{forecast_id}")
async def get_forecast_detail(forecast_id: str, current_user: dict = Depends(get_current_user), db = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    forecast = await db["forecasts"].find_one({"_id": forecast_id, "user_id": current_user["id"]})
    if not forecast:
        raise HTTPException(status_code=404, detail="Forecast not found")
    return forecast

@router.get("/alerts")
async def get_alerts(current_user: dict = Depends(get_current_user), db = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    alerts = await db["alerts"].find({"user_id": current_user["id"]}).sort("timestamp", -1).to_list(100)
    return alerts

@router.patch("/alerts/{alert_id}")
async def update_alert_status(alert_id: str, update: AlertUpdate, current_user: dict = Depends(get_current_user), db = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    if update.status not in ["new", "investigating", "resolved"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    result = await db["alerts"].update_one(
        {"_id": alert_id, "user_id": current_user["id"]},
        {"$set": {"status": update.status}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found or status not changed")
    return {"message": "Alert updated successfully"}

@router.get("/network-states")
async def get_network_states(current_user: dict = Depends(get_current_user), db = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    states = await db["network_states"].find({"user_id": current_user["id"]}).sort("timestamp", -1).limit(50).to_list(50)
    return states
