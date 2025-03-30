# stock_detector.py - Core module for real-time stock detection

import cv2
import numpy as np
import time
import os
import threading
import queue
import logging
from datetime import datetime
import requests
import json
from typing import Dict, List, Tuple, Optional
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StockDetector")

class StockDetector:
    def __init__(self, 
                camera_id: int = 0, 
                camera_url: str = None,
                model_path: str = "models/food_detection_yolov8n.pt",
                confidence_threshold: float = 0.4,
                api_endpoint: str = "http://localhost:8000",
                detection_interval: int = 30,  # seconds
                notification_threshold: Dict[str, int] = None,
                enable_weight_sensor: bool = False,
                weight_sensor_port: str = None,
                ):
        """
        Initialize the Stock Detector
        
        Args:
            camera_id: Index of the camera (for local webcams)
            camera_url: URL of IP camera stream (if using IP camera)
            model_path: Path to YOLOv8 model
            confidence_threshold: Minimum confidence score for detections
            api_endpoint: Backend API endpoint
            detection_interval: Time between stock level checks (seconds)
            notification_threshold: Dictionary mapping item names to minimum quantity thresholds
            enable_weight_sensor: Whether to use weight sensors
            weight_sensor_port: Serial port for weight sensor communication
        """
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.api_endpoint = api_endpoint
        self.detection_interval = detection_interval
        self.notification_threshold = notification_threshold or {}
        self.enable_weight_sensor = enable_weight_sensor
        self.weight_sensor_port = weight_sensor_port
        
        # Initialize video capture
        self.cap = None
        self.initialize_camera()
        
        # Load YOLOv8 model
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            self.model.conf = confidence_threshold
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
        
        # Initialize weight sensor if enabled
        self.weight_sensor = None
        if self.enable_weight_sensor:
            self.initialize_weight_sensor()
        
        # State variables
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.inventory_state = {}  # Current inventory state
        self.reference_sizes = {}  # Reference sizes for items
        self.last_detection_time = time.time()
        
        # Threads
        self.capture_thread = None
        self.detection_thread = None
        self.notification_thread = None
    
    def initialize_camera(self):
        """Initialize the camera capture"""
        try:
            if self.camera_url:
                self.cap = cv2.VideoCapture(self.camera_url)
                logger.info(f"Initialized IP camera at {self.camera_url}")
            else:
                self.cap = cv2.VideoCapture(self.camera_id)
                logger.info(f"Initialized local camera (ID: {self.camera_id})")
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return True
        
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def initialize_weight_sensor(self):
        """Initialize the weight sensor (placeholder for actual implementation)"""
        if not self.weight_sensor_port:
            logger.warning("Weight sensor enabled but no port specified")
            return False
        
        try:
            # This is a placeholder for actual sensor integration
            # In a real implementation, you would:
            # 1. Import appropriate library for your sensor (e.g., pyserial for Arduino)
            # 2. Establish communication with the sensor
            # 3. Setup calibration
            
            logger.info(f"Initialized weight sensor on port {self.weight_sensor_port}")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing weight sensor: {e}")
            return False
    
    def start(self):
        """Start the stock detection system"""
        if self.running:
            logger.warning("Stock detector is already running")
            return
        
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                logger.error("Cannot start stock detector: camera not available")
                return
        
        if not self.model:
            logger.error("Cannot start stock detector: model not loaded")
            return
        
        # Initialize state
        self.running = True
        self.fetch_current_inventory()
        
        # Start threads
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.notification_thread = threading.Thread(target=self.notification_loop)
        
        self.capture_thread.daemon = True
        self.detection_thread.daemon = True
        self.notification_thread.daemon = True
        
        self.capture_thread.start()
        self.detection_thread.start()
        self.notification_thread.start()
        
        logger.info("Stock detector started")
    
    def stop(self):
        """Stop the stock detection system"""
        self.running = False
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        if self.notification_thread:
            self.notification_thread.join(timeout=2.0)
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Stock detector stopped")
    
    def capture_loop(self):
        """Thread that captures frames from the camera"""
        last_frame_time = time.time()
        
        while self.running:
            try:
                # Read frame from camera
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(1)
                    continue
                
                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - last_frame_time)
                last_frame_time = current_time
                
                # Add frame to queue if not full
                if not self.frame_queue.full():
                    self.frame_queue.put((frame, current_time))
                else:
                    # Skip frame if queue is full
                    _ = self.frame_queue.get_nowait()
                    self.frame_queue.put((frame, current_time))
                
                # Control capture rate
                time.sleep(0.01)  # ~100 FPS max
            
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(1)
    
    def detection_loop(self):
        """Thread that processes frames and detects stock levels"""
        while self.running:
            current_time = time.time()
            
            # Check if it's time to do another stock detection
            if current_time - self.last_detection_time >= self.detection_interval:
                try:
                    # Get the most recent frame
                    if not self.frame_queue.empty():
                        frame, timestamp = self.frame_queue.get()
                        
                        # Process frame
                        stock_levels = self.detect_stock_levels(frame)
                        
                        # Add weight sensor data if available
                        if self.enable_weight_sensor:
                            weight_data = self.read_weight_sensors()
                            self.combine_detection_with_weight(stock_levels, weight_data)
                        
                        # Update results
                        self.result_queue.put((stock_levels, timestamp))
                        
                        # Update detection time
                        self.last_detection_time = current_time
                        
                        # Update inventory if changes detected
                        self.update_inventory(stock_levels)
                
                except Exception as e:
                    logger.error(f"Error in detection loop: {e}")
            
            # Don't use 100% CPU
            time.sleep(0.1)
    
    def notification_loop(self):
        """Thread that handles notifications for low stock levels"""
        while self.running:
            try:
                # Check result queue for new detections
                if not self.result_queue.empty():
                    stock_levels, timestamp = self.result_queue.get()
                    
                    # Check if any items are below threshold
                    low_stock_items = []
                    for item_name, item_data in stock_levels.items():
                        threshold = self.notification_threshold.get(item_name, 1)
                        if item_data['quantity'] <= threshold:
                            low_stock_items.append({
                                'name': item_name,
                                'quantity': item_data['quantity'],
                                'threshold': threshold
                            })
                    
                    # Send notifications for low stock items
                    if low_stock_items:
                        self.send_notifications(low_stock_items)
            
            except Exception as e:
                logger.error(f"Error in notification loop: {e}")
            
            # Check every few seconds
            time.sleep(5)
    
    def detect_stock_levels(self, frame):
        """
        Detect stock levels from a frame
        
        Args:
            frame: The image frame to process
        
        Returns:
            Dict of detected items with quantity estimates
        """
        # Apply object detection
        results = self.model(frame)
        
        # Process results
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        
        # Initialize stock levels dictionary
        stock_levels = {}
        
        # Process each detection
        for detection in detections:
            item_name = detection["name"]
            confidence = float(detection["confidence"])
            
            # Extract bounding box
            x_min = float(detection["xmin"])
            y_min = float(detection["ymin"])
            x_max = float(detection["xmax"])
            y_max = float(detection["ymax"])
            
            # Calculate area of bounding box
            area = (x_max - x_min) * (y_max - y_min)
            
            # If we already have this item, update it based on larger area
            if item_name in stock_levels:
                if area > stock_levels[item_name]["area"]:
                    stock_levels[item_name] = {
                        "area": area,
                        "bbox": (x_min, y_min, x_max, y_max),
                        "confidence": confidence
                    }
            else:
                stock_levels[item_name] = {
                    "area": area,
                    "bbox": (x_min, y_min, x_max, y_max),
                    "confidence": confidence
                }
        
        # Estimate quantities from areas
        self.estimate_quantities(stock_levels, frame)
        
        return stock_levels
    
    def estimate_quantities(self, stock_levels, frame):
        """
        Estimate quantities of items based on visual data
        
        Args:
            stock_levels: Dictionary of detected items
            frame: The original frame
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        frame_area = height * width
        
        for item_name, item_data in stock_levels.items():
            # Get the bounding box
            x_min, y_min, x_max, y_max = item_data["bbox"]
            
            # Create a mask for the object region
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 255
            
            # Apply contour detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            
            # Apply threshold
            _, thresh = cv2.threshold(masked_gray, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If we found contours, use them to refine area estimation
            if contours:
                contour_area = sum(cv2.contourArea(contour) for contour in contours)
                # Use contour area if it's reasonable
                if contour_area > 0 and contour_area < item_data["area"]:
                    refined_area = contour_area
                else:
                    refined_area = item_data["area"]
            else:
                refined_area = item_data["area"]
            
            # Save refined area
            item_data["refined_area"] = refined_area
            
            # Get reference size if available
            if item_name in self.reference_sizes:
                reference_area = self.reference_sizes[item_name]
                # Calculate quantity based on area ratio
                quantity = round(refined_area / reference_area)
            else:
                # First time seeing this item - set as reference
                self.reference_sizes[item_name] = refined_area
                quantity = 1
                logger.info(f"Set reference size for {item_name}: {refined_area}")
            
            # Add quantity to item data (ensure at least 1)
            item_data["quantity"] = max(1, quantity)
    
    def read_weight_sensors(self):
        """Read data from weight sensors (placeholder for actual implementation)"""
        # This is a placeholder for actual weight sensor reading
        # In a real implementation, you would:
        # 1. Read data from your connected sensors
        # 2. Process and return the weight data
        
        # Placeholder return value
        return {}
    
    def combine_detection_with_weight(self, stock_levels, weight_data):
        """
        Combine computer vision detection with weight sensor data
        
        Args:
            stock_levels: Dictionary of detected items
            weight_data: Dictionary of weight sensor readings
        """
        # Placeholder for actual implementation
        # In a real implementation, you would:
        # 1. Match weight data with detected items based on shelf position
        # 2. Refine quantity estimates using weight data
        
        # No actual implementation for now
        pass
    
    def fetch_current_inventory(self):
        """Fetch current inventory from the backend API"""
        try:
            response = requests.get(f"{self.api_endpoint}/inventory/")
            
            if response.status_code == 200:
                data = response.json()
                
                # Update inventory state
                self.inventory_state = {item["name"]: item for item in data.get("inventory", [])}
                logger.info(f"Fetched current inventory: {len(self.inventory_state)} items")
                
                return True
            else:
                logger.error(f"Failed to fetch inventory: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"Error fetching inventory: {e}")
            return False
    
    def update_inventory(self, stock_levels):
        """
        Update inventory based on detected stock levels
        
        Args:
            stock_levels: Dictionary of detected items with quantity estimates
        """
        items_to_update = []
        
        for item_name, item_data in stock_levels.items():
            quantity = item_data["quantity"]
            
            # Check if item exists in inventory and if quantity changed
            if item_name in self.inventory_state:
                current_quantity = self.inventory_state[item_name].get("quantity", 0)
                
                # Only update if quantity changed significantly
                if abs(quantity - current_quantity) >= 1:
                    # Get freshness status from spoilage detection
                    freshness_status = self.get_freshness_status(item_name, item_data.get("image"))
                    spoilage_score = self.get_spoilage_score(item_name, item_data.get("image"))
                    
                    items_to_update.append({
                        "name": item_name,
                        "quantity": quantity,
                        "category": self.inventory_state[item_name].get("category", "unknown"),
                        "freshness_status": freshness_status,
                        "spoilage_score": spoilage_score
                    })
            else:
                # New item
                freshness_status = self.get_freshness_status(item_name, item_data.get("image"))
                spoilage_score = self.get_spoilage_score(item_name, item_data.get("image"))
                
                items_to_update.append({
                    "name": item_name,
                    "quantity": quantity,
                    "category": "unknown",
                    "freshness_status": freshness_status,
                    "spoilage_score": spoilage_score
                })
        
        # Update inventory in database
        if items_to_update:
            self.update_database(items_to_update)
    
    def get_freshness_status(self, item_name, image):
        """Get freshness status of an item based on visual analysis"""
        if not image:
            return "unknown"
        
        # Convert image to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get item config if available
        item_config = self.get_item_config(item_name)
        if not item_config:
            return "unknown"
        
        # Analyze color ranges
        color_ranges = item_config.get("color_ranges", {})
        if not color_ranges:
            return "unknown"
        
        # Check each color range
        for status, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            percentage = np.sum(mask > 0) / (image.shape[0] * image.shape[1])
            
            if percentage > 0.5:  # If more than 50% of pixels match the color range
                return status
        
        return "unknown"
    
    def get_spoilage_score(self, item_name, image):
        """Calculate spoilage score for an item"""
        if not image:
            return 0.0
        
        # Get item config if available
        item_config = self.get_item_config(item_name)
        if not item_config:
            return 0.0
        
        # Convert image to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture score using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_threshold = item_config.get("texture_threshold", 0.5)
        
        # Combine color and texture analysis
        color_score = self.get_color_score(image, item_config)
        texture_score = 1.0 - min(1.0, laplacian_var / (texture_threshold * 1000))
        
        # Weight the scores
        final_score = 0.7 * color_score + 0.3 * texture_score
        
        return min(1.0, max(0.0, final_score))
    
    def get_color_score(self, image, item_config):
        """Calculate color-based spoilage score"""
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
    
    def get_item_config(self, item_name):
        """Get configuration for a specific item"""
        # Load config if not already loaded
        if not hasattr(self, 'config'):
            self.load_spoilage_config()
        
        # Search for item in config
        for category in self.config.get("spoilage_thresholds", {}):
            if item_name in self.config["spoilage_thresholds"][category]:
                return self.config["spoilage_thresholds"][category][item_name]
        
        return None
    
    def send_notifications(self, low_stock_items):
        """
        Send notifications for low stock items
        
        Args:
            low_stock_items: List of items with low stock
        """
        if not low_stock_items:
            return
        
        # Format notification message
        message = "Low stock alert:\n"
        for item in low_stock_items:
            message += f"- {item['name']}: {item['quantity']} left (threshold: {item['threshold']})\n"
        
        logger.info(f"Sending notification: {message}")
        
        # Send notification (implementations below)
        self.send_email_notification(message)
        self.send_sms_notification(message)
        self.send_app_notification(message)
    
    def send_email_notification(self, message):
        """Send email notification (placeholder implementation)"""
        # This is a placeholder for actual email sending
        # In a real implementation, you would:
        # 1. Use SMTP or an email service API (e.g., SendGrid, Mailgun)
        # 2. Format and send the email
        
        logger.info(f"Email notification would be sent: {message}")
    
    def send_sms_notification(self, message):
        """Send SMS notification (placeholder implementation)"""
        # This is a placeholder for actual SMS sending
        # In a real implementation, you would:
        # 1. Use an SMS service API (e.g., Twilio, Nexmo)
        # 2. Format and send the SMS
        
        logger.info(f"SMS notification would be sent: {message}")
    
    def send_app_notification(self, message):
        """Send mobile app notification (placeholder implementation)"""
        # This is a placeholder for actual push notification
        # In a real implementation, you would:
        # 1. Use a push notification service (e.g., Firebase Cloud Messaging)
        # 2. Format and send the notification
        
        logger.info(f"App notification would be sent: {message}")

# Main function for standalone execution
def main():
    # Create and start the stock detector
    detector = StockDetector(
        camera_id=0,  # Use first camera
        model_path="models/food_detection_yolov8n.pt",
        detection_interval=30,  # Check every 30 seconds
        notification_threshold={
            "apple": 2,
            "milk": 1,
            "bread": 1,
            # Add more items and thresholds as needed
        }
    )
    
    detector.start()
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Stopping stock detector...")
    
    finally:
        detector.stop()

if __name__ == "__main__":
    main()