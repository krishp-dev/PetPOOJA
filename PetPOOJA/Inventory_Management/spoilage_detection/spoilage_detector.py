# spoilage_detector.py - Core module for food spoilage detection

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from datetime import datetime
import json
import logging
import requests
from pathlib import Path
import threading
import queue
from typing import Dict, List, Tuple, Optional
import serial
import board
import adafruit_dht
import adafruit_sht31d

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spoilage_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SpoilageDetector")

class SpoilageDetector:
    def __init__(self, 
                camera_id: int = 0,
                camera_url: str = None,
                model_path: str = "models/spoilage_detection_model.h5",
                yolo_model_path: str = "models/yolov8_spoilage.pt",
                api_endpoint: str = "http://localhost:8000",
                detection_interval: int = 3600,  # Check once per hour by default
                enable_sensors: bool = False,
                temp_humidity_pin: int = 4,
                ):
        """
        Initialize the Food Spoilage Detector
        
        Args:
            camera_id: Index of the camera (for local webcams)
            camera_url: URL of IP camera stream (if using IP camera)
            model_path: Path to CNN classification model for spoilage detection
            yolo_model_path: Path to YOLOv8 model for spoilage region detection
            api_endpoint: Backend API endpoint
            detection_interval: Time between spoilage checks (seconds)
            enable_sensors: Whether to use temperature/humidity sensors
            temp_humidity_pin: GPIO pin for DHT sensor
        """
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.model_path = model_path
        self.yolo_model_path = yolo_model_path
        self.api_endpoint = api_endpoint
        self.detection_interval = detection_interval
        self.enable_sensors = enable_sensors
        self.temp_humidity_pin = temp_humidity_pin
        
        # Initialize video capture
        self.cap = None
        self.initialize_camera()
        
        # Load models
        self.cnn_model = self.load_cnn_model()
        self.yolo_model = self.load_yolo_model()
        
        # Initialize sensors
        self.temp_humidity_sensor = None
        if self.enable_sensors:
            self.initialize_sensors()
        
        # State variables
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.inventory_state = {}  # Current inventory state
        self.last_detection_time = time.time()
        
        # Threads
        self.capture_thread = None
        self.detection_thread = None
        self.notification_thread = None
        
        # Load food spoilage thresholds
        self.load_spoilage_config()
    
    def load_spoilage_config(self):
        """Load food spoilage configuration"""
        config_path = "config/spoilage_config.json"
        
        try:
            if not os.path.exists(config_path):
                # Create default config
                default_config = {
                    "spoilage_thresholds": {
                        "fruit": {
                            "banana": {
                                "shelf_life_days": 5,
                                "color_ranges": {
                                    "fresh": [[20, 100, 100], [35, 255, 255]],  # Yellow HSV
                                    "medium": [[15, 100, 100], [20, 255, 255]],  # Yellow-Brown HSV
                                    "spoiled": [[0, 100, 100], [15, 255, 255]]   # Brown HSV
                                },
                                "texture_threshold": 0.7
                            },
                            "apple": {
                                "shelf_life_days": 14,
                                "color_ranges": {
                                    "fresh": [[0, 100, 100], [10, 255, 255]],    # Red HSV
                                    "medium": [[10, 50, 100], [20, 255, 255]],   # Red-Brown HSV
                                    "spoiled": [[20, 50, 50], [30, 255, 255]]    # Brown HSV
                                },
                                "texture_threshold": 0.6
                            }
                        },
                        "vegetable": {
                            "tomato": {
                                "shelf_life_days": 7,
                                "color_ranges": {
                                    "fresh": [[0, 100, 100], [10, 255, 255]],    # Red HSV
                                    "warning": [[10, 100, 100], [20, 255, 255]],  # Orange-Red HSV
                                    "spoiled": [[20, 50, 50], [30, 255, 255]]     # Brown HSV
                                },
                                "texture_threshold": 0.6
                            },
                            "lettuce": {
                                "shelf_life_days": 7,
                                "color_ranges": {
                                    "fresh": [[35, 50, 50], [85, 255, 255]],     # Green HSV
                                    "warning": [[25, 50, 50], [35, 255, 255]],    # Yellow-Green HSV
                                    "spoiled": [[20, 50, 50], [30, 255, 255]]     # Brown HSV
                                },
                                "texture_threshold": 0.5
                            },
                            "carrot": {
                                "shelf_life_days": 14,
                                "color_ranges": {
                                    "fresh": [[10, 100, 100], [20, 255, 255]],   # Orange HSV
                                    "warning": [[20, 100, 100], [30, 255, 255]],  # Orange-Brown HSV
                                    "spoiled": [[30, 50, 50], [40, 255, 255]]     # Brown HSV
                                },
                                "texture_threshold": 0.7
                            },
                            "onion": {
                                "shelf_life_days": 30,
                                "color_ranges": {
                                    "fresh": [[0, 0, 200], [180, 30, 255]],      # White/Pale HSV
                                    "warning": [[0, 30, 100], [20, 150, 200]],    # Light Brown HSV
                                    "spoiled": [[0, 50, 50], [30, 255, 150]]      # Dark Brown HSV
                                },
                                "texture_threshold": 0.6
                            },
                            "potato": {
                                "shelf_life_days": 30,
                                "color_ranges": {
                                    "fresh": [[0, 0, 200], [180, 30, 255]],      # White/Pale HSV
                                    "warning": [[0, 30, 100], [20, 150, 200]],    # Light Brown HSV
                                    "spoiled": [[0, 50, 50], [30, 255, 150]]      # Dark Brown HSV
                                },
                                "texture_threshold": 0.7
                            }
                        },
                        "dairy": {
                            "milk": {
                                "shelf_life_days": 7,
                                "temperature_threshold": 4.0  # Celsius
                            },
                            "cheese": {
                                "shelf_life_days": 14,
                                "mold_threshold": 0.3,
                                "temperature_threshold": 7.0  # Celsius
                            }
                        },
                        "meat": {
                            "chicken": {
                                "shelf_life_days": 2,
                                "color_ranges": {
                                    "fresh": [[0, 0, 200], [180, 30, 255]],      # White/Pink HSV
                                    "medium": [[0, 30, 100], [20, 150, 200]],    # Light Brown HSV
                                    "spoiled": [[0, 50, 50], [30, 255, 150]]     # Dark Brown HSV
                                },
                                "temperature_threshold": 4.0  # Celsius
                            },
                            "beef": {
                                "shelf_life_days": 3,
                                "color_ranges": {
                                    "fresh": [[0, 100, 100], [10, 255, 255]],    # Red HSV
                                    "medium": [[10, 100, 100], [20, 255, 255]],  # Dark Red HSV
                                    "spoiled": [[20, 50, 50], [40, 255, 150]]    # Brown/Gray HSV
                                },
                                "temperature_threshold": 4.0  # Celsius
                            }
                        }
                    },
                    "environmental_factors": {
                        "optimal_temperature": {
                            "fridge": [0.0, 4.0],  # Celsius
                            "freezer": [-18.0, -15.0],  # Celsius
                            "pantry": [15.0, 21.0]  # Celsius
                        },
                        "optimal_humidity": {
                            "fridge": [80, 90],  # Percent
                            "freezer": [30, 50],  # Percent
                            "pantry": [50, 60]  # Percent
                        }
                    },
                    "notification_settings": {
                        "alert_days_before_expiry": 2,
                        "critical_alert_threshold": 0.8  # Spoilage confidence threshold for critical alerts
                    }
                }
                
                # Save default config
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                
                self.config = default_config
                logger.info(f"Created default spoilage configuration at {config_path}")
            else:
                # Load existing config
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded spoilage configuration from {config_path}")
        
        except Exception as e:
            logger.error(f"Error loading spoilage config: {e}")
            self.config = {}
    
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
    
    def load_cnn_model(self):
        """Load CNN model for spoilage classification"""
        try:
            if os.path.exists(self.model_path):
                model = load_model(self.model_path)
                logger.info(f"Loaded CNN model from {self.model_path}")
                return model
            else:
                logger.warning(f"CNN model file not found: {self.model_path}")
                return None
        
        except Exception as e:
            logger.error(f"Error loading CNN model: {e}")
            return None
    
    def load_yolo_model(self):
        """Load YOLOv8 model for spoilage detection"""
        try:
            if os.path.exists(self.yolo_model_path):
                import torch
                
                model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path=self.yolo_model_path)
                model.conf = 0.4  # Confidence threshold
                logger.info(f"Loaded YOLO model from {self.yolo_model_path}")
                return model
            else:
                logger.warning(f"YOLO model file not found: {self.yolo_model_path}")
                return None
        
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            return None
    
    def initialize_sensors(self):
        """Initialize temperature and humidity sensors"""
        try:
            if self.enable_sensors:
                # Try to initialize DHT22 sensor first
                try:
                    self.temp_humidity_sensor = adafruit_dht.DHT22(self.temp_humidity_pin)
                    logger.info(f"Initialized DHT22 sensor on pin {self.temp_humidity_pin}")
                except:
                    # If DHT22 fails, try SHT31D over I2C
                    try:
                        i2c = board.I2C()
                        self.temp_humidity_sensor = adafruit_sht31d.SHT31D(i2c)
                        logger.info("Initialized SHT31D sensor over I2C")
                    except Exception as e:
                        logger.error(f"Failed to initialize both DHT22 and SHT31D sensors: {e}")
                        self.temp_humidity_sensor = None
        
        except Exception as e:
            logger.error(f"Error initializing temperature/humidity sensors: {e}")
            self.temp_humidity_sensor = None
    
    def start(self):
        """Start the spoilage detection system"""
        if self.running:
            logger.warning("Spoilage detector is already running")
            return
        
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                logger.error("Cannot start spoilage detector: camera not available")
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
        
        logger.info("Spoilage detector started")
    
    def stop(self):
        """Stop the spoilage detection system"""
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
        
        logger.info("Spoilage detector stopped")
    
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
                
                # Add frame to queue if not full
                if not self.frame_queue.full():
                    self.frame_queue.put((frame, time.time()))
                else:
                    # Skip frame if queue is full
                    _ = self.frame_queue.get_nowait()
                    self.frame_queue.put((frame, time.time()))
                
                # Control capture rate (1 FPS is enough for spoilage detection)
                time.sleep(1.0)
            
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(1)
    
    def detection_loop(self):
        """Thread that processes frames and detects food spoilage"""
        while self.running:
            current_time = time.time()
            
            # Check if it's time to do another spoilage detection
            if current_time - self.last_detection_time >= self.detection_interval:
                try:
                    # Get the most recent frame
                    if not self.frame_queue.empty():
                        frame, timestamp = self.frame_queue.get()
                        
                        # Read temperature and humidity data if sensors are available
                        env_data = self.read_sensors() if self.enable_sensors else None
                        
                        # Process frame for spoilage detection
                        spoilage_results = self.detect_spoilage(frame, env_data)
                        
                        # Add results to queue
                        self.result_queue.put((spoilage_results, timestamp))
                        
                        # Update detection time
                        self.last_detection_time = current_time
                        
                        # Update inventory with spoilage information
                        self.update_inventory_spoilage(spoilage_results)
                
                except Exception as e:
                    logger.error(f"Error in detection loop: {e}")
            
            # Don't use 100% CPU
            time.sleep(10)
    
    def notification_loop(self):
        """Thread that handles notifications for spoiled or near-expiry items"""
        while self.running:
            try:
                # Check result queue for new detections
                if not self.result_queue.empty():
                    spoilage_results, timestamp = self.result_queue.get()
                    
                    # Find items that need notification
                    spoiled_items = []
                    near_expiry_items = []
                    
                    for item_name, item_data in spoilage_results.items():
                        spoilage_score = item_data.get("spoilage_score", 0.0)
                        days_remaining = item_data.get("days_remaining", 0)
                        
                        # Critical alert threshold from config
                        critical_threshold = self.config.get("notification_settings", {}).get("critical_alert_threshold", 0.8)
                        alert_days = self.config.get("notification_settings", {}).get("alert_days_before_expiry", 2)
                        
                        # Check if item is spoiled or near expiry
                        if spoilage_score >= critical_threshold:
                            spoiled_items.append({
                                "name": item_name,
                                "spoilage_score": spoilage_score,
                                "issues": item_data.get("issues", [])
                            })
                        elif days_remaining <= alert_days:
                            near_expiry_items.append({
                                "name": item_name,
                                "days_remaining": days_remaining
                            })
                    
                    # Send notifications
                    if spoiled_items:
                        self.send_spoilage_alert(spoiled_items)
                    
                    if near_expiry_items:
                        self.send_expiry_alert(near_expiry_items)
                        self.suggest_recipes(near_expiry_items)
            
            except Exception as e:
                logger.error(f"Error in notification loop: {e}")
            
            # Check every minute
            time.sleep(60)
    
    def read_sensors(self):
        """Read temperature and humidity data from sensors"""
        if not self.temp_humidity_sensor:
            return None
        
        try:
            if isinstance(self.temp_humidity_sensor, adafruit_dht.DHT22):
                temperature = self.temp_humidity_sensor.temperature
                humidity = self.temp_humidity_sensor.humidity
            else:
                temperature = self.temp_humidity_sensor.temperature
                humidity = self.temp_humidity_sensor.relative_humidity
            
            return {
                "temperature": temperature,
                "humidity": humidity,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error reading sensor data: {e}")
            return None
    
    def detect_spoilage(self, frame, env_data=None):
        """
        Detect food spoilage in an image frame
        
        Args:
            frame: The image frame to process
            env_data: Environmental data (temperature, humidity)
        
        Returns:
            Dictionary with spoilage detection results for each item
        """
        # Results dictionary
        results = {}
        
        # 1. Use YOLO to detect food items and potential spoilage regions
        if self.yolo_model:
            yolo_results = self.yolo_model(frame)
            detections = yolo_results.pandas().xyxy[0].to_dict(orient="records")
            
            # Process each detection
            for detection in detections:
                item_name = detection["name"]
                confidence = float(detection["confidence"])
                
                # Extract bounding box
                x_min, y_min, x_max, y_max = map(int, [
                    detection["xmin"], 
                    detection["ymin"], 
                    detection["xmax"], 
                    detection["ymax"]
                ])
                
                # Crop the region for the detected item
                item_image = frame[y_min:y_max, x_min:x_max]
                
                # Skip if cropped image is too small
                if item_image.size == 0 or item_image.shape[0] < 10 or item_image.shape[1] < 10:
                    continue
                
                # Process the item for spoilage
                item_result = self.analyze_item_spoilage(item_name, item_image, env_data)
                
                # Add to results
                results[item_name] = item_result
        
        # 2. If no YOLO model or no detections, use CNN on the whole frame
        if not results and self.cnn_model:
            # Preprocess image for CNN
            img = cv2.resize(frame, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            prediction = self.cnn_model.predict(img)[0]
            
            # If using a binary classification (fresh vs spoiled)
            spoilage_score = float(prediction[0])
            
            # Add a generic result
            results["unknown_food"] = {
                "spoilage_score": spoilage_score,
                "confidence": 0.5,
                "days_remaining": 0 if spoilage_score > 0.7 else 1,
                "issues": ["Potential spoilage detected"]
            }
        
        # 3. For items in inventory that weren't detected in the image
        # Add estimated spoilage based on expiry dates
        for item_name, item_data in self.inventory_state.items():
            if item_name not in results:
                # Calculate days remaining based on purchase date and shelf life
                if "purchase_date" in item_data:
                    purchase_date = datetime.fromisoformat(item_data["purchase_date"])
                    days_since_purchase = (datetime.now() - purchase_date).days
                    
                    # Get shelf life from config or use default
                    category = item_data.get("category", "unknown")
                    shelf_life = 7  # Default shelf life
                    
                    if category in self.config.get("spoilage_thresholds", {}):
                        if item_name in self.config["spoilage_thresholds"][category]:
                            shelf_life = self.config["spoilage_thresholds"][category][item_name].get("shelf_life_days", 7)
                    
                    days_remaining = max(0, shelf_life - days_since_purchase)
                    spoilage_score = min(1.0, max(0.0, days_since_purchase / shelf_life))
                    
                    results[item_name] = {
                        "spoilage_score": spoilage_score,
                        "confidence": 0.7,
                        "days_remaining": days_remaining,
                        "issues": ["Estimated based on purchase date"]
                    }
        
        return results
    
    def analyze_item_spoilage(self, item_name, item_image, env_data=None):
        """
        Analyze a specific food item for signs of spoilage
        
        Args:
            item_name: Name of the food item
            item_image: Cropped image of the food item
            env_data: Environmental data (temperature, humidity)
        
        Returns:
            Dictionary with spoilage analysis results
        """
        # Initialize result with default values
        result = {
            "spoilage_score": 0.0,
            "confidence": 0.0,
            "days_remaining": 7,
            "issues": []
        }
        
        # Get current date
        current_date = datetime.now().date()
        
        # Look up the item in inventory to get purchase date
        if item_name in self.inventory_state:
            item_data = self.inventory_state[item_name]
            
            # If we have purchase date, calculate days since purchase
            if "purchase_date" in item_data:
                purchase_date = datetime.fromisoformat(item_data["purchase_date"]).date()
                days_since_purchase = (current_date - purchase_date).days
                result["days_since_purchase"] = days_since_purchase
        
        # Try to find the item and its category in the spoilage config
        found_in_config = False
        for category, items in self.config.get("spoilage_thresholds", {}).items():
            if item_name in items:
                item_config = items[item_name]
                found_in_config = True
                
                # Get shelf life
                shelf_life = item_config.get("shelf_life_days", 7)
                
                # If we know days since purchase, calculate days remaining
                if "days_since_purchase" in result:
                    days_remaining = max(0, shelf_life - result["days_since_purchase"])
                    result["days_remaining"] = days_remaining
                    
                    # Calculate base spoilage score from days remaining
                    base_spoilage = 1.0 - (days_remaining / shelf_life)
                    result["spoilage_score"] = max(0.0, min(1.0, base_spoilage))
                
                # Check visual cues for spoilage
                visual_score = self.analyze_visual_spoilage(item_image, item_config)
                
                # Check environmental factors
                env_score = 0.0
                if env_data and "temperature_threshold" in item_config:
                    temp = env_data["temperature"]
                    threshold = item_config["temperature_threshold"]
                    
                    # Higher temperature increases spoilage risk
                    if temp > threshold:
                        temp_factor = min(1.0, (temp - threshold) / 10.0)
                        env_score = max(env_score, temp_factor)
                        result["issues"].append(f"Temperature too high: {temp:.1f}°C")
                
                # Combine scores (visual cues have higher weight)
                final_score = max(result["spoilage_score"], visual_score)
                if env_score > 0:
                    final_score = min(1.0, final_score * (1.0 + env_score * 0.5))
                
                result["spoilage_score"] = final_score
                result["confidence"] = 0.8  # Higher confidence with config
                
                break
        
        # If item wasn't found in config, use general visual analysis
        if not found_in_config:
            # Use CNN model if available
            if self.cnn_model:
                # Preprocess image for CNN
                img = cv2.resize(item_image, (224, 224))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)
                
                # Make prediction
                prediction = self.cnn_model.predict(img)[0]
                
                # If using a binary classification (fresh vs spoiled)
                spoilage_score = float(prediction[0])
                
                result["spoilage_score"] = spoilage_score
                result["confidence"] = 0.6
                result["days_remaining"] = max(0, int(7 * (1 - spoilage_score)))
                
                if spoilage_score > 0.7:
                    result["issues"].append("High spoilage risk detected")
            else:
                # Basic visual analysis without model
                result["issues"].append("Item not in database, using visual estimation only")
                
                # Simple color analysis
                hsv = cv2.cvtColor(item_image, cv2.COLOR_BGR2HSV)
                
                # Detect common spoilage colors (brown, black, green-mold)
                brown_mask = cv2.inRange(hsv, np.array([10, 100, 20]), np.array([20, 255, 200]))
                black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 30]))
                green_mold_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
                
                # Calculate percentage of spoilage colors
                total_pixels = item_image.shape[0] * item_image.shape[1]
                brown_percent = np.sum(brown_mask > 0) / total_pixels
                black_percent = np.sum(black_mask > 0) / total_pixels
                mold_percent = np.sum(green_mold_mask > 0) / total_pixels
                
                # Combine percentages with weights
                spoilage_score = min(1.0, (brown_percent * 0.4 + black_percent * 0.8 + mold_percent * 1.0) * 5)
                
                if black_percent > 0.05:
                    result["issues"].append("Dark spots detected")
                
                if mold_percent > 0.02:
                    result["issues"].append("Possible mold detected")
                
                result["spoilage_score"] = spoilage_score
                result["confidence"] = 0.5
                result["days_remaining"] = max(0, int(7 * (1 - spoilage_score)))
        
        # Add visual classification result
        if result["spoilage_score"] < 0.3:
            result["classification"] = "fresh"
        elif result["spoilage_score"] < 0.7:
            result["classification"] = "medium"
        else:
            result["classification"] = "spoiled"
        
        return result
    
    def analyze_visual_spoilage(self, image, item_config):
        """
        Analyze visual cues for food spoilage
        
        Args:
            image: Image of the food item
            item_config: Configuration for the specific food item
        
        Returns:
            Spoilage score based on visual analysis (0.0 to 1.0)
        """
        # Initialize variables
        spoilage_score = 0.0
        issues = []
        
        # Convert to HSV for color analysis
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Check color ranges if available in config
        if "color_ranges" in item_config:
            color_ranges = item_config["color_ranges"]
            
            # Check for spoiled color range
            if "spoiled" in color_ranges:
                lower, upper = color_ranges["spoiled"]
                mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
                spoiled_percent = np.sum(mask > 0) / (image.shape[0] * image.shape[1])
                
                if spoiled_percent > 0.2:
                    spoilage_score = max(spoilage_score, spoiled_percent * 2)
                    issues.append("Spoilage color detected")
            
            # Check for medium color range
            if "medium" in color_ranges and spoilage_score < 0.5:
                lower, upper = color_ranges["medium"]
                mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
                medium_percent = np.sum(mask > 0) / (image.shape[0] * image.shape[1])
                
                if medium_percent > 0.3:
                    medium_score = medium_percent * 0.8
                    spoilage_score = max(spoilage_score, medium_score)
                    issues.append("Medium spoilage color detected")
            
            # Check for fresh color range
            if "fresh" in color_ranges and spoilage_score < 0.3:
                lower, upper = color_ranges["fresh"]
                mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
                fresh_percent = np.sum(mask > 0) / (image.shape[0] * image.shape[1])
                
                if fresh_percent > 0.5:
                    spoilage_score = max(spoilage_score, 0.1)
                    issues.append("Fresh color detected")
        
        # Add texture analysis if available
        if "texture_threshold" in item_config:
            texture_threshold = item_config["texture_threshold"]
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            
            if laplacian_var < texture_threshold:
                spoilage_score = max(spoilage_score, 0.7)
                issues.append("Texture indicates spoilage")
        
        # Return the final spoilage score
        return spoilage_score