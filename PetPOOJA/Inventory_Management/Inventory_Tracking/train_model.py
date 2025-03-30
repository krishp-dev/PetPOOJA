# train_model.py

import os
import argparse
import yaml
from ultralytics import YOLO
import torch
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_dataset_yaml(data_dir, yaml_path):
    """Create a YAML file for dataset configuration"""
    
    # Ensure paths exist
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        logger.error(f"Training or validation directory not found in {data_dir}")
        return False
    
    # Get class names from directory
    class_names = []
    labels_path = os.path.join(data_dir, "labels.txt")
    
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Try to infer from directory structure
        for filename in os.listdir(train_dir):
            if os.path.isdir(os.path.join(train_dir, filename)):
                class_names.append(filename)
    
    if not class_names:
        logger.error("No class names found. Please provide a labels.txt file.")
        return False
    
    # Create YAML configuration
    yaml_data = {
        'path': os.path.abspath(data_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',  # Optional
        'nc': len(class_names),
        'names': class_names
    }
    
    # Write YAML file
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    logger.info(f"Created dataset configuration at {yaml_path}")
    return True

def train_yolo_model(data_yaml, model_size='n', epochs=100, batch_size=16, image_size=640, output_dir='runs/train'):
    """Train a YOLOv8 model on the dataset"""
    
    # Check if CUDA is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load a pre-trained model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            project=output_dir,
            name=f'food_detection_yolov8{model_size}',
            device=device,
            patience=20,  # Early stopping patience
            save=True,  # Save checkpoints
            pretrained=True,  # Use pretrained weights
            optimizer='Adam',  # Optimizer
            lr0=0.001,  # Initial learning rate
            cos_lr=True,  # Use cosine learning rate scheduler
        )
        
        logger.info(f"Training completed. Results saved to {results.save_dir}")
        
        # Validate the model
        val_results = model.val()
        logger.info(f"Validation results: {val_results}")
        
        return model, results
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return None, None

def export_model(model, format='onnx', output_path=None):
    """Export the trained model to the specified format"""
    try:
        if output_path is None:
            output_path = f'models/food_detection.{format}'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export the model
        success = model.export(format=format, save=True)
        
        if success:
            logger.info(f"Model exported successfully to {output_path}")
            return True
        else:
            logger.error(f"Failed to export model to {format}")
            return False
    
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        return False

def main():
    """Main function to train and export the model"""
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for food detection')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], 
                        help='YOLOv8 model size (n, s, m, l, x)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--image_size', type=int, default=640, help='Input image size')
    parser.add_argument('--output_dir', type=str, default='runs/train', help='Output directory for training results')
    parser.add_argument('--export_format', type=str, default='onnx', 
                        choices=['onnx', 'torchscript', 'tflite', 'coreml', 'saved_model'], 
                        help='Format to export the model to')
    
    args = parser.parse_args()
    
    # Create dataset YAML
    yaml_path = os.path.join(args.data_dir, 'dataset.yaml')
    if not create_dataset_yaml(args.data_dir, yaml_path):
        return
    
    # Train the model
    model, results = train_yolo_model(
        yaml_path, 
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        output_dir=args.output_dir
    )
    
    if model is None:
        return
    
    # Export the model
    output_path = os.path.join('models', f'food_detection_yolov8{args.model_size}.{args.export_format}')
    export_model(model, format=args.export_format, output_path=output_path)

if __name__ == "__main__":
    main()