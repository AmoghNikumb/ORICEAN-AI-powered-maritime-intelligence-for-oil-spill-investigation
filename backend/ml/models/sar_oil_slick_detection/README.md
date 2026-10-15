---
license: apache-2.0
tags:
- pytorch
- sar
- oil-slick-detection
- maritime-monitoring
- environmental-monitoring
- satellite-imagery
- resnet
- computer-vision
language:
- en
metrics:
- accuracy
- precision
- recall
datasets:
- sentinel-1-sar
task_categories:
- image-classification
- object-detection
widget:
- example_title: "SAR Image Analysis"
  text: "Upload a Sentinel-1 SAR image to detect oil slicks"
model-index:
- name: SAR Oil Slick Detection
  results:
  - task:
      type: image-classification
      name: Oil Slick Detection
    dataset:
      type: sentinel-1-sar
      name: Sentinel-1 SAR Images
    metrics:
    - type: accuracy
      value: 0.948
      name: Validation Accuracy
    - type: accuracy
      value: 0.851
      name: Training Accuracy
---

# SAR Oil Slick Detection Model

## Model Description

This model is a ResNet-based deep learning model trained to detect oil slicks in Synthetic Aperture Radar (SAR) satellite imagery. It's part of an end-to-end AI pipeline for maritime monitoring and illegal discharge detection.

## Pipeline Integration

This model is integrated into a comprehensive maritime monitoring system:

1. **Data Ingestion**: Airflow scheduler ingests ship AIS data every few minutes
2. **Anomaly Detection**: AIS anomaly model flags suspicious ship behavior
3. **Satellite Data**: For flagged zones, Sentinel-1 radar imagery is automatically fetched (±12 hours)
4. **Preprocessing**: Automated preprocessing using pyroSAR and snappy (no manual SNAP GUI required)
5. **Oil Slick Detection**: This SAR ML model outputs oil slick masks with confidence scores
6. **Spatio-temporal Fusion**: If oil slick and ship anomaly overlap within 10km and ±12h, ML Alert is issued
7. **Real-time Dashboard**: Alerts are pushed to Streamlit dashboard with live map and severity index

## Model Architecture

- **Base Model**: ResNet (Convolutional Neural Network)
- **Task**: Binary classification/segmentation for oil slick detection
- **Input**: SAR satellite imagery (224x224 pixels)
- **Output**: Oil slick masks with confidence scores

## Training Details

- **Batch Size**: 8
- **Epochs**: 15
- **Learning Rate**: 0.001
- **Image Size**: 224x224 pixels
- **Training Samples**: 1,574
- **Validation Samples**: 1,615

## Performance

Based on training history:
- **Final Training Loss**: ~0.34
- **Final Validation Loss**: ~0.19
- **Final Training Accuracy**: ~85%
- **Final Validation Accuracy**: ~95%

## Files Included

- `model.pth`: PyTorch model weights
- `training_history.pkl`: Complete training metrics and hyperparameters
- `config.json`: Model configuration
- `inference.py`: Sample inference script

## Usage

```python
import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = torch.load('model.pth', map_location='cpu')
model.eval()

# Load preprocessing transforms (adjust based on your training setup)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Adjust for SAR imagery
])

# Inference
def predict_oil_slick(image_path):
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        confidence = torch.sigmoid(output).item()
    
    return confidence

# Example usage
confidence_score = predict_oil_slick('sar_image.png')
print(f"Oil slick confidence: {confidence_score:.4f}")
```

## Data Sources

- **Sentinel-1 SAR Data**: European Space Agency's Sentinel-1 satellite constellation
- **Preprocessing**: pyroSAR and snappy for automated SAR data processing

## Applications

- Maritime pollution monitoring
- Illegal oil discharge detection
- Environmental compliance enforcement
- Ocean pollution assessment
- Real-time maritime surveillance

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{sar_oil_slick_model,
  title={SAR Oil Slick Detection Model for Maritime Monitoring},
  author={Meghana K},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/MeghanaK25/sar-oil-slick-detection}
}
```

## License

This model is released under the Apache 2.0 License.

## Contact

For questions or collaborations regarding this model or the maritime monitoring pipeline, please reach out through the Hugging Face model page or create an issue in the repository.

---

**Note**: This model is designed for research and environmental monitoring purposes. For production deployment in critical maritime surveillance applications, additional validation and testing may be required.
