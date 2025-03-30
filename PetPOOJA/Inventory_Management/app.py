# app.py - Main FastAPI Backend

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import torch
from pymongo import MongoClient
import datetime
import os
from typing import List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Visual Inventory Tracking API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
try:
    mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    db = mongo_client["inventory_db"]
    inventory_collection = db["inventory"]
    logger.info("MongoDB connection established")
except Exception as e:
    logger.error(f"MongoDB connection error: {e}")

# Load YOLOv8 model - using a pre-trained model or your custom model
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', 
                          path=os.getenv("MODEL_PATH", "models/food_detection_model.pt"))
    model.conf = 0.4  # Confidence threshold
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading error: {e}")
    model = None

@app.get("/")
async def root():
    return {"message": "Visual Inventory Tracking API is running"}

@app.post("/detect-items/")
async def detect_items(file: UploadFile = File(...)):
    """Endpoint to detect food items in an uploaded image"""
    
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and process the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform inference
        results = model(img)
        
        # Process results
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        
        # Extract detected items
        detected_items = []
        for detection in detections:
            item = {
                "name": detection["name"],
                "confidence": float(detection["confidence"]),
                "bounding_box": {
                    "x_min": float(detection["xmin"]),
                    "y_min": float(detection["ymin"]),
                    "x_max": float(detection["xmax"]),
                    "y_max": float(detection["ymax"])
                }
            }
            detected_items.append(item)
        
        return JSONResponse(content={"detected_items": detected_items})
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/update-inventory/")
async def update_inventory(items: List[dict]):
    """Endpoint to update inventory with detected items"""
    try:
        timestamp = datetime.datetime.now()
        
        # Prepare inventory update
        for item in items:
            # Check if item exists in inventory
            existing_item = inventory_collection.find_one({"name": item["name"]})
            
            if existing_item:
                # Update quantity
                new_quantity = existing_item.get("quantity", 0) + item.get("quantity", 1)
                inventory_collection.update_one(
                    {"name": item["name"]},
                    {"$set": {
                        "quantity": new_quantity,
                        "last_updated": timestamp
                    }}
                )
            else:
                # Add new item
                inventory_collection.insert_one({
                    "name": item["name"],
                    "quantity": item.get("quantity", 1),
                    "category": item.get("category", "unknown"),
                    "created_at": timestamp,
                    "last_updated": timestamp
                })
        
        return {"message": f"Successfully updated inventory with {len(items)} items"}
    
    except Exception as e:
        logger.error(f"Error updating inventory: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating inventory: {str(e)}")

@app.get("/inventory/")
async def get_inventory():
    """Endpoint to retrieve current inventory"""
    try:
        items = list(inventory_collection.find({}, {"_id": 0}))
        return {"inventory": items}
    
    except Exception as e:
        logger.error(f"Error retrieving inventory: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving inventory: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)