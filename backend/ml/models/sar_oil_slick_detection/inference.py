import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import json

class SARToilSlickDetector:
    """
    SAR Oil Slick Detection Model for Maritime Monitoring
    
    This model detects oil slicks in Synthetic Aperture Radar (SAR) satellite imagery
    and is part of an end-to-end maritime monitoring pipeline.
    """
    
    def __init__(self, model_path='model.pth', config_path='config.json'):
        """
        Initialize the oil slick detector
        
        Args:
            model_path: Path to the PyTorch model file
            config_path: Path to the configuration JSON file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load model
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Setup preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1) if self.config['architecture']['num_channels'] == 1 else transforms.ToTensor(),
            transforms.ToTensor() if self.config['architecture']['num_channels'] == 1 else lambda x: x,
            transforms.Normalize(
                mean=self.config['data']['normalization']['mean'],
                std=self.config['data']['normalization']['std']
            )
        ])
    
    def preprocess_image(self, image_input):
        """
        Preprocess input image for model inference
        
        Args:
            image_input: PIL Image, numpy array, or path to image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image_input, str):
            # Load from file path
            image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            # Already a PIL Image
            image = image_input
        else:
            raise ValueError("Input must be a file path, PIL Image, or numpy array")
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return input_tensor
    
    def predict(self, image_input, return_confidence=True):
        """
        Predict oil slick presence in SAR image
        
        Args:
            image_input: PIL Image, numpy array, or path to image file
            return_confidence: Whether to return confidence score or binary prediction
            
        Returns:
            float or bool: Confidence score (0-1) or binary prediction
        """
        input_tensor = self.preprocess_image(image_input)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Apply sigmoid to get confidence score
            if len(output.shape) > 1 and output.shape[1] > 1:
                # Multi-class output, use softmax
                confidence = F.softmax(output, dim=1)[:, 1].item()  # Oil slick class
            else:
                # Binary output, use sigmoid
                confidence = torch.sigmoid(output).item()
        
        if return_confidence:
            return confidence
        else:
            threshold = self.config['deployment']['recommended_threshold']
            return confidence > threshold
    
    def batch_predict(self, image_list, batch_size=8):
        """
        Predict oil slick presence for a batch of images
        
        Args:
            image_list: List of images (PIL Images, numpy arrays, or file paths)
            batch_size: Batch size for processing
            
        Returns:
            list: List of confidence scores
        """
        results = []
        
        for i in range(0, len(image_list), batch_size):
            batch = image_list[i:i+batch_size]
            batch_tensors = []
            
            for image in batch:
                tensor = self.preprocess_image(image)
                batch_tensors.append(tensor.squeeze(0))
            
            # Stack tensors into batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    # Multi-class output
                    confidences = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                else:
                    # Binary output
                    confidences = torch.sigmoid(outputs).cpu().numpy()
                
                results.extend(confidences.tolist())
        
        return results

def load_training_history(history_path='training_history.pkl'):
    """
    Load and display training history
    
    Args:
        history_path: Path to training history pickle file
        
    Returns:
        dict: Training history dictionary
    """
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    print("Training History Summary:")
    print(f"Number of epochs: {len(history['train_losses'])}")
    print(f"Final training loss: {history['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {history['val_losses'][-1]:.4f}")
    print(f"Final training accuracy: {history['train_accuracies'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_accuracies'][-1]:.4f}")
    print(f"Hyperparameters: {history['hyperparameters']}")
    
    return history

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = SARToilSlickDetector()
    
    # Load training history
    history = load_training_history()
    
    # Example prediction (replace with actual SAR image path)
    # confidence = detector.predict('path_to_sar_image.tiff')
    # print(f"Oil slick confidence: {confidence:.4f}")
    
    # Example batch prediction
    # image_paths = ['image1.tiff', 'image2.tiff', 'image3.tiff']
    # confidences = detector.batch_predict(image_paths)
    # for i, conf in enumerate(confidences):
    #     print(f"Image {i+1} oil slick confidence: {conf:.4f}")
    
    print("Model loaded successfully and ready for inference!")
