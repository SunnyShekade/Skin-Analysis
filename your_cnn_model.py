import numpy as np
import cv2  # OpenCV for image processing
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.nn.functional as F

# Load your pre-trained CNN model for skin analysis
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Example model, replace with your own
num_classes = 35  # Ensure this matches your model's output
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.eval()  # Set the model to evaluation mode

# Define the image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load OpenCV Haar cascade for face detection
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

def preprocess_image(image_file):
    image_file.seek(0)
    img_np = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img_np is None:
        raise ValueError("Image could not be decoded. Please upload a valid image.")
    return img_np

def detect_human_face(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

def safe_argmax(prob_slice):
    if prob_slice.numel() == 0:
        return 0
    return torch.argmax(prob_slice).item()

def map_category_with_dynamic_percentage(index, probabilities, categories):
    if not categories or index >= len(categories):
        return ("Unknown", 0)
    category = categories[index]
    probability_values = [p.item() for p in probabilities]
    total_prob = sum(probability_values)
    if total_prob > 0:
        normalized_probs = [p / total_prob for p in probability_values]
    else:
        normalized_probs = [0] * len(probability_values)
    percentage = normalized_probs[index] * 100
    if percentage < 10:
        percentage = 10
    return category, round(percentage, 4)

def map_skin_tone(index, probabilities):
    categories = ['Light', 'Medium', 'Dark']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_acne_level(index, probabilities):
    categories = ['None', 'Mild', 'Moderate', 'Severe']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_blackheads(index, probabilities):
    categories = ['None', 'Few', 'Moderate', 'Many']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_dark_circles(index, probabilities):
    categories = ['None', 'Mild', 'Moderate', 'Severe']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_skin_type(index, probabilities):
    categories = ['Oily', 'Dry', 'Combination']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_hair_quality(index, probabilities):
    categories = ['Poor', 'Average', 'Good']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_hydration(index, probabilities):
    categories = ['Dehydrated', 'Normal', 'Well-Hydrated']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_sensitivity(index, probabilities):
    categories = ['Low', 'Medium', 'High']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_wrinkles(index, probabilities):
    categories = ['None', 'Few', 'Moderate', 'Many']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_pore_size(index, probabilities):
    categories = ['Small', 'Medium', 'Large']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def analyze(image_file):
    img_np = preprocess_image(image_file)

    human_found = detect_human_face(img_np)
    if not human_found:
        return {
            "living": False,
            "skinMetrics": None,
            "recommendedProducts": []
        }

    img_t = transform(img_np).unsqueeze(0)  # batch dim

    with torch.no_grad():
        outputs = model(img_t)
        probabilities = F.softmax(outputs, dim=1)

    print("Probabilities shape:", probabilities.shape)

    if outputs.numel() == 0:
        raise ValueError("Model output is empty. Please check the input image and model.")

    if probabilities.shape[1] < 35:
        raise ValueError("Model output does not have the expected number of classes.")

    def slice_or_empty(tensor, start, end):
        if end <= tensor.size(0) and start < end:
            return tensor[start:end]
        return torch.tensor([])

    tone_slice = slice_or_empty(probabilities[0], 0, 3)
    acne_slice = slice_or_empty(probabilities[0], 3, 7)
    blackheads_slice = slice_or_empty(probabilities[0], 7, 11)
    dark_circles_slice = slice_or_empty(probabilities[0], 11, 15)
    skin_type_slice = slice_or_empty(probabilities[0], 15, 19)
    hair_quality_slice = slice_or_empty(probabilities[0], 19, 22)
    hydration_slice = slice_or_empty(probabilities[0], 22, 25)
    sensitivity_slice = slice_or_empty(probabilities[0], 25, 28)
    wrinkles_slice = slice_or_empty(probabilities[0], 28, 32)
    pore_size_slice = slice_or_empty(probabilities[0], 32, 35)

    tone_idx = safe_argmax(tone_slice)
    acne_idx = safe_argmax(acne_slice)
    blackheads_idx = safe_argmax(blackheads_slice)
    dark_circles_idx = safe_argmax(dark_circles_slice)
    skin_type_idx = safe_argmax(skin_type_slice)
    hair_quality_idx = safe_argmax(hair_quality_slice)
    hydration_idx = safe_argmax(hydration_slice)
    sensitivity_idx = safe_argmax(sensitivity_slice)
    wrinkles_idx = safe_argmax(wrinkles_slice)
    pore_size_idx = safe_argmax(pore_size_slice)

    skin_metrics = {
        'tone': map_skin_tone(tone_idx, tone_slice),
        'acne_level': map_acne_level(acne_idx, acne_slice),
        'blackheads': map_blackheads(blackheads_idx, blackheads_slice),
        'dark_circles': map_dark_circles(dark_circles_idx, dark_circles_slice),
        'skin_type': map_skin_type(skin_type_idx, skin_type_slice),
        'hair_quality': map_hair_quality(hair_quality_idx, hair_quality_slice),
        'hydration_level': map_hydration(hydration_idx, hydration_slice),
        'sensitivity': map_sensitivity(sensitivity_idx, sensitivity_slice),
        'wrinkles': map_wrinkles(wrinkles_idx, wrinkles_slice),
        'pore_size': map_pore_size(pore_size_idx, pore_size_slice),
    }

    print("Skin Metrics:", skin_metrics)

    return {
        "living": True,
        "skinMetrics": skin_metrics,
        "recommendedProducts": []
    }
