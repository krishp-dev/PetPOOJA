# raspi_setup.py - Raspberry Pi setup and IoT sensor integration
import time
import serial
import board
import busio
import adafruit_vl53l0x
import adafruit_hx711
import RPi.GPIO as GPIO
import threading
import json
import requests
import logging
from pathlib import Path
import subprocess
import os
from datetime import datetime
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("raspi_iot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RaspiIoT")

class IoTSensorHub:
    def __init__(self, 
                config_path="config/sensor_config.json",
                api_endpoint="http://localhost:8000",
                enable_camera=True,
                enable_weight_sensors=True,
                enable_distance_sensors=True,
                enable_barcode_scanner=True,
                data_upload_interval=60,  # seconds
                ):
        """
        Initialize the IoT Sensor Hub for ingredient tracking
        
        Args:
            config_path: Path to sensor configuration file
            api_endpoint: Backend API endpoint URL
            enable_camera: Whether to enable the camera
            enable_weight_sensors: Whether to enable weight sensors
            enable_distance_sensors: Whether to enable distance sensors
            enable_barcode_scanner: Whether to enable barcode scanner
            data_upload_interval: How often to upload data to server (seconds)
        """
        self.config_path = config_path
        self.api_endpoint = api_endpoint
        self.enable_camera = enable_camera
        self.enable_weight_sensors = enable_weight_sensors
        self.enable_distance_sensors = enable_distance_sensors
        self.enable_barcode_scanner = enable_barcode_scanner
        self.data_upload_interval = data_upload_interval
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize sensors
        self.weight_sensors = {}
        self.distance_sensors = {}
        self.barcode_scanner = None
        
        # Initialize camera stream process
        self.camera_process = None
        
        # Initialize sensor data
        self.sensor_data = {
            "weight": {},
            "distance": {},
            "barcodes": [],
            "timestamp": None
        }
        
        # State variables
        self.running = False
        self.last_upload_time = 0
        
        # Threads
        self.sensor_thread = None
        self.barcode_thread = None
    
    def load_config(self):
        """Load sensor configuration from JSON file"""
        try:
            config_file = Path(self.config_path)
            
            if not config_file.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                return self.create_default_config()
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.create_default_config()
    
    def create_default_config(self):
        """Create default configuration"""
        default_config = {
            "weight_sensors": [
                {
                    "name": "shelf_1",
                    "dout_pin": 5,
                    "sck_pin": 6,
                    "reference_unit": 1000,
                    "offset": 0,
                    "item_mapping": "milk"
                },
                {
                    "name": "shelf_2",
                    "dout_pin": 17,
                    "sck_pin": 18,
                    "reference_unit": 1000,
                    "offset": 0,
                    "item_mapping": "apples"
                }
            ],
            "distance_sensors": [
                {
                    "name": "bin_1",
                    "i2c_address": 0x29,
                    "item_mapping": "flour",
                    "container_height": 30  # cm
                },
                {
                    "name": "bin_2",
                    "i2c_address": 0x30,
                    "item_mapping": "sugar",
                    "container_height": 25  # cm
                }
            ],
            "barcode_scanner": {
                "type": "usb",
                "port": "/dev/ttyACM0",
                "baud_rate": 9600
            },
            "camera": {
                "resolution": [1280, 720],
                "framerate": 15,
                "rotation": 0,
                "stream_port": 8080
            }
        }
        
        # Save default config
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            
            logger.info(f"Created default configuration at {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving default config: {e}")
        
        return default_config
    
    def setup_gpio(self):
        """Setup GPIO for sensors"""
        try:
            # Set GPIO mode
            GPIO.setmode(GPIO.BCM)
            logger.info("GPIO initialized")
            return True
        
        except Exception as e:
            logger.error(f"Error setting up GPIO: {e}")
            return False
    
    def setup_weight_sensors(self):
        """Setup HX711 weight sensors"""
        if not self.enable_weight_sensors:
            return
        
        try:
            for sensor_config in self.config.get("weight_sensors", []):
                name = sensor_config["name"]
                dout_pin = sensor_config["dout_pin"]
                sck_pin = sensor_config["sck_pin"]
                reference_unit = sensor_config["reference_unit"]
                offset = sensor_config["offset"]
                
                # Initialize HX711 sensor
                hx = adafruit_hx711.HX711(sck_pin, dout_pin)
                
                # Set calibration values
                hx.set_scale(reference_unit)
                hx.reset()
                hx.tare()
                
                # Store sensor instance
                self.weight_sensors[name] = {
                    "sensor": hx,
                    "config": sensor_config
                }
                
                logger.info(f"Initialized weight sensor: {name}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error setting up weight sensors: {e}")
            return False
    
    def setup_distance_sensors(self):
        """Setup VL53L0X ToF distance sensors"""
        if not self.enable_distance_sensors:
            return
        
        try:
            # Initialize I2C bus
            i2c = busio.I2C(board.SCL, board.SDA)
            
            for sensor_config in self.config.get("distance_sensors", []):
                name = sensor_config["name"]
                i2c_address = sensor_config["i2c_address"]
                
                # Initialize VL53L0X sensor
                sensor = adafruit_vl53l0x.VL53L0X(i2c)
                sensor.set_address(i2c_address)
                
                # Store sensor instance
                self.distance_sensors[name] = {
                    "sensor": sensor,
                    "config": sensor_config
                }
                
                logger.info(f"Initialized distance sensor: {name}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error setting up distance sensors: {e}")
            return False
    
    def setup_barcode_scanner(self):
        """Setup barcode scanner"""
        if not self.enable_barcode_scanner:
            return
        
        try:
            barcode_config = self.config.get("barcode_scanner", {})
            
            if barcode_config["type"] == "usb":
                # Initialize serial connection to barcode scanner
                self.barcode_scanner = serial.Serial(
                    port=barcode_config["port"],
                    baudrate=barcode_config["baud_rate"],
                    timeout=1
                )
                
                logger.info(f"Initialized barcode scanner on {barcode_config['port']}")
                return True
            else:
                logger.error(f"Unsupported barcode scanner type: {barcode_config['type']}")
                return False
        
        except Exception as e:
            logger.error(f"Error setting up barcode scanner: {e}")
            return False
    
    def setup_camera_stream(self):
        """Setup Raspberry Pi camera stream"""
        if not self.enable_camera:
            return
        
        try:
            camera_config = self.config.get("camera", {})
            width, height = camera_config.get("resolution", [1280, 720])
            framerate = camera_config.get("framerate", 15)
            rotation = camera_config.get("rotation", 0)
            port = camera_config.get("stream_port", 8080)
            
            # Start camera stream using UV4L (separate process)
            cmd = [
                "raspivid",
                "-o", "-",                           # Output to stdout
                "-t", "0",                           # No timeout
                "-w", str(width),                    # Width
                "-h", str(height),                   # Height
                "-fps", str(framerate),              # Framerate
                "-rot", str(rotation),               # Rotation
                "-b", "2000000",                     # Bitrate
                "-n",                                # No preview
                "|", "cvlc", "stream:///dev/stdin",  # Pipe to VLC
                "--sout", f"'#standard{{{port}}}'",  # Stream on specified port
                "--sout-keep"                        # Keep streaming
            ]
            
            # Start the process
            command = " ".join(cmd)
            self.camera_process = subprocess.Popen(command, shell=True)
            
            logger.info(f"Started camera stream on port {port}")
            return True
        
        except Exception as e:
            logger.error(f"Error setting up camera stream: {e}")
            return False
    
    def read_weight_sensors(self):
        """Read data from all weight sensors"""
        weight_data = {}
        
        for name, sensor_info in self.weight_sensors.items():
            try:
                # Read weight from the sensor
                sensor = sensor_info["sensor"]
                weight = sensor.get_weight(5)  # Average over 5 readings
                
                # Apply offset and store the value
                offset = sensor_info["config"].get("offset", 0)
                weight_data[name] = max(0, weight - offset)  # Ensure no negative weights
                
                logger.info(f"Weight sensor {name}: {weight_data[name]} grams")
            except Exception as e:
                logger.error(f"Error reading weight sensor {name}: {e}")
        
        return weight_data

    def read_distance_sensors(self):
        """Read data from all distance sensors"""
        distance_data = {}
        
        for name, sensor_info in self.distance_sensors.items():
            try:
                # Read distance from the sensor
                sensor = sensor_info["sensor"]
                distance = sensor.range
                
                # Store the value
                distance_data[name] = distance
                logger.info(f"Distance sensor {name}: {distance} mm")
            except Exception as e:
                logger.error(f"Error reading distance sensor {name}: {e}")
        
        return distance_data

    def read_barcode_scanner(self):
        """Read data from the barcode scanner"""
        barcodes = []
        
        if self.barcode_scanner:
            try:
                while self.barcode_scanner.in_waiting > 0:
                    barcode = self.barcode_scanner.readline().decode('utf-8').strip()
                    barcodes.append(barcode)
                    logger.info(f"Scanned barcode: {barcode}")
            except Exception as e:
                logger.error(f"Error reading barcode scanner: {e}")
        
        return barcodes

    def upload_data(self):
        """Upload sensor data to the backend API"""
        try:
            self.sensor_data["timestamp"] = datetime.now().isoformat()
            response = requests.post(self.api_endpoint, json=self.sensor_data)
            
            if response.status_code == 200:
                logger.info("Data uploaded successfully")
            else:
                logger.error(f"Failed to upload data: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error uploading data: {e}")

    def start(self):
        """Start the IoT Sensor Hub"""
        self.running = True
        self.last_upload_time = time.time()
        
        # Start sensor reading thread
        self.sensor_thread = threading.Thread(target=self.run_sensors)
        self.sensor_thread.start()
        
        # Start barcode scanner thread
        if self.enable_barcode_scanner:
            self.barcode_thread = threading.Thread(target=self.run_barcode_scanner)
            self.barcode_thread.start()
        
        logger.info("IoT Sensor Hub started")

    def stop(self):
        """Stop the IoT Sensor Hub"""
        self.running = False
        
        # Stop threads
        if self.sensor_thread:
            self.sensor_thread.join()
        if self.barcode_thread:
            self.barcode_thread.join()
        
        # Stop camera stream
        if self.camera_process:
            self.camera_process.terminate()
            self.camera_process = None
        
        # Cleanup GPIO
        GPIO.cleanup()
        
        logger.info("IoT Sensor Hub stopped")

    def run_sensors(self):
        """Thread to read sensors and upload data periodically"""
        while self.running:
            try:
                # Read sensors
                if self.enable_weight_sensors:
                    self.sensor_data["weight"] = self.read_weight_sensors()
                if self.enable_distance_sensors:
                    self.sensor_data["distance"] = self.read_distance_sensors()
                
                # Upload data periodically
                current_time = time.time()
                if current_time - self.last_upload_time >= self.data_upload_interval:
                    self.upload_data()
                    self.last_upload_time = current_time
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in sensor thread: {e}")

    def run_barcode_scanner(self):
        """Thread to read barcodes continuously"""
        while self.running:
            try:
                barcodes = self.read_barcode_scanner()
                self.sensor_data["barcodes"].extend(barcodes)
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in barcode scanner thread: {e}")
    def cleanup(self):
            """Cleanup resources before exiting"""
            try:
                # Stop the IoT Sensor Hub
                self.stop()
    
                # Close barcode scanner connection if open
                if self.barcode_scanner and self.barcode_scanner.is_open:
                    self.barcode_scanner.close()
    
                logger.info("Resources cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
                # Placeholder code is empty; no modifications needed here.
if __name__ == "__main__":
    try:
        # Initialize IoT Sensor Hub
        sensor_hub = IoTSensorHub()

        # Setup components
        if sensor_hub.setup_gpio():
            logger.info("GPIO setup completed")

        if sensor_hub.enable_weight_sensors and sensor_hub.setup_weight_sensors():
            logger.info("Weight sensors setup completed")

        if sensor_hub.enable_distance_sensors and sensor_hub.setup_distance_sensors():
            logger.info("Distance sensors setup completed")

        if sensor_hub.enable_barcode_scanner and sensor_hub.setup_barcode_scanner():
            logger.info("Barcode scanner setup completed")

        if sensor_hub.enable_camera and sensor_hub.setup_camera_stream():
            logger.info("Camera stream setup completed")

        # Start the IoT Sensor Hub
        sensor_hub.start()

        # Keep the program running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Cleanup resources
        if 'sensor_hub' in locals():
            sensor_hub.cleanup()