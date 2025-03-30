import cv2
import numpy as np
import json
import logging
from typing import Tuple, Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("freshness_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FreshnessDetector")

class FreshnessDetector:
    def __init__(self, config_path: str = "config/spoilage_config.json"):
        """Initialize the Freshness Detector"""
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load spoilage configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def analyze_freshness(self, item_name: str, image: np.ndarray) -> Tuple[str, float]:
        """Analyze freshness of a food item"""
        try:
            # Convert image to HSV for color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Get item config
            item_config = self.get_item_config(item_name)
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
                    spoilage_score = self.calculate_spoilage_score(image, item_config)
                    return status, spoilage_score
            
            return "unknown", 0.0
        
        except Exception as e:
            logger.error(f"Error analyzing freshness: {e}")
            return "unknown", 0.0
    
    def calculate_spoilage_score(self, image: np.ndarray, item_config: Dict) -> float:
        """Calculate spoilage score for an item"""
        try:
            # Convert image to grayscale for texture analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture score using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            texture_threshold = item_config.get("texture_threshold", 0.5)
            
            # Combine color and texture analysis
            color_score = self.calculate_color_score(image, item_config)
            texture_score = 1.0 - min(1.0, laplacian_var / (texture_threshold * 1000))
            
            # Weight the scores
            final_score = 0.7 * color_score + 0.3 * texture_score
            
            return min(1.0, max(0.0, final_score))
        
        except Exception as e:
            logger.error(f"Error calculating spoilage score: {e}")
            return 0.0
    
    def calculate_color_score(self, image: np.ndarray, item_config: Dict) -> float:
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
    
    def get_item_config(self, item_name: str) -> Optional[Dict]:
        """Get configuration for a specific item"""
        try:
            # Search for item in config
            for category in self.config.get("spoilage_thresholds", {}):
                if item_name in self.config["spoilage_thresholds"][category]:
                    return self.config["spoilage_thresholds"][category][item_name]
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting item config: {e}")
            return None 