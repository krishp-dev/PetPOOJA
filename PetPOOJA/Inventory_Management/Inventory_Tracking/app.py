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
import json
from freshness_detector import FreshnessDetector

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

# Initialize freshness detector
freshness_detector = FreshnessDetector()

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
        
        # Extract detected items with freshness analysis
        detected_items = []
        for detection in detections:
            # Get bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, [
                detection["xmin"],
                detection["ymin"],
                detection["xmax"],
                detection["ymax"]
            ])
            
            # Crop the item image
            item_image = img[y_min:y_max, x_min:x_max]
            
            # Analyze freshness
            freshness_status, spoilage_score = freshness_detector.analyze_freshness(
                detection["name"],
                item_image
            )
            
            item = {
                "name": detection["name"],
                "confidence": float(detection["confidence"]),
                "bounding_box": {
                    "x_min": float(detection["xmin"]),
                    "y_min": float(detection["ymin"]),
                    "x_max": float(detection["xmax"]),
                    "y_max": float(detection["ymax"])
                },
                "freshness_status": freshness_status,
                "spoilage_score": spoilage_score
            }
            detected_items.append(item)
        
        return JSONResponse(content={"detected_items": detected_items})
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def analyze_freshness(item_name, image):
    """Analyze freshness of a food item"""
    try:
        # Convert image to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get item config
        item_config = get_item_config(item_name)
        if not item_config:
            return "unknown", 0.0
        
        # Analyze color ranges
        color_ranges = item_config.get("color_ranges", {})
        if not color_ranges:
            return "unknown", 0.0
        
        # Check each color range
        for status, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            percentage = np.sum(mask > 0) / (image.shape[0] * image.shape[1])
            
            if percentage > 0.5:  # If more than 50% of pixels match the color range
                # Calculate spoilage score
                spoilage_score = calculate_spoilage_score(image, item_config)
                return status, spoilage_score
        
        return "unknown", 0.0
    
    except Exception as e:
        logger.error(f"Error analyzing freshness: {e}")
        return "unknown", 0.0

def calculate_spoilage_score(image, item_config):
    """Calculate spoilage score for an item"""
    try:
        # Convert image to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture score using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_threshold = item_config.get("texture_threshold", 0.5)
        
        # Combine color and texture analysis
        color_score = calculate_color_score(image, item_config)
        texture_score = 1.0 - min(1.0, laplacian_var / (texture_threshold * 1000))
        
        # Weight the scores
        final_score = 0.7 * color_score + 0.3 * texture_score
        
        return min(1.0, max(0.0, final_score))
    
    except Exception as e:
        logger.error(f"Error calculating spoilage score: {e}")
        return 0.0

def calculate_color_score(image, item_config):
    """Calculate color-based spoilage score"""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_ranges = item_config.get("color_ranges", {})
        
        if not color_ranges:
            return 0.0
        
        # Calculate percentage of pixels in each color range
        scores = {}
        for status, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            percentage = np.sum(mask > 0) / (image.shape[0] * image.shape[1])
            scores[status] = percentage
        
        # Weight the scores based on status
        weights = {
            "fresh": 0.0,
            "warning": 0.5,
            "spoiled": 1.0
        }
        
        # Calculate weighted average
        total_score = sum(scores.get(status, 0) * weights.get(status, 0) 
                         for status in weights)
        
        return min(1.0, max(0.0, total_score))
    
    except Exception as e:
        logger.error(f"Error calculating color score: {e}")
        return 0.0

def get_item_config(item_name):
    """Get configuration for a specific item"""
    try:
        # Load config if not already loaded
        if not hasattr(get_item_config, 'config'):
            config_path = "config/spoilage_config.json"
            with open(config_path, 'r') as f:
                get_item_config.config = json.load(f)
        
        # Search for item in config
        for category in get_item_config.config.get("spoilage_thresholds", {}):
            if item_name in get_item_config.config["spoilage_thresholds"][category]:
                return get_item_config.config["spoilage_thresholds"][category][item_name]
        
        return None
    
    except Exception as e:
        logger.error(f"Error getting item config: {e}")
        return None

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
                    "freshness_status": item.get("freshness_status", "unknown"),  # fresh, warning, or spoiled
                    "spoilage_score": item.get("spoilage_score", 0.0),  # 0.0 to 1.0
                    "last_detection": timestamp,
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