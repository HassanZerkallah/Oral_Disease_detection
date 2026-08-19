
# ============================================
# ORAL DISEASE CLASSIFICATION APP
# EfficientNet-B0 + Streamlit
# ============================================
 
# --------------------------------------------
# IMPORT REQUIRED LIBRARIES
# --------------------------------------------
 
import streamlit as st
 
import torch
import torch.nn as nn
 
from torchvision import models, transforms
 
from PIL import Image
 
from pathlib import Path
 
 
# ============================================
# PAGE CONFIGURATION
# ============================================
 
# Configure the Streamlit page.
st.set_page_config(
    page_title="Oral Disease Classifier",
    page_icon="🦷",
    layout="centered"
)
 
 
# ============================================
# MODEL CONFIGURATION
# ============================================
 
# Number of classes in our trained model.
NUM_CLASSES = 6
 
# Image size used during training.
IMAGE_SIZE = 224
 
# Define the six disease classes in exactly
# the same order used during training.
CLASS_NAMES = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Mouth Ulcer",
    "Tooth Discoloration"
]
 
 
# ============================================
# DEVICE
# ============================================
 
# Use GPU when CUDA is available.
# Otherwise automatically fall back to CPU.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
 
 
# ============================================
# MODEL PATH
# ============================================
 
# Build the path to the saved EfficientNet
# checkpoint relative to this app.py file.
MODEL_PATH = (
    Path(__file__).parent
 
    / "efficientnet_b0_best.pth"
)
 
 
# ============================================
# IMAGE PREPROCESSING
# ============================================
 
# These are the same ImageNet statistics
# used during validation and testing.
IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]
 
IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]
 
 
# Validation/test preprocessing.
# IMPORTANT:
# We do NOT use random augmentation during
# prediction.
transform = transforms.Compose([
 
    # Resize the uploaded image to the same
    # size used during model training.
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
 
    # Convert the image to a PyTorch tensor.
    transforms.ToTensor(),
 
    # Normalize using ImageNet statistics.
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])
 
 
# ============================================
# LOAD MODEL
# ============================================
 
@st.cache_resource
def load_model():
 
    # Create EfficientNet-B0 architecture.
    model = models.efficientnet_b0(
        weights=None
    )
 
    # Get the number of inputs to the original
    # ImageNet classification layer.
    num_features = (
        model.classifier[-1].in_features
    )
 
    # Replace the original 1000-class layer
    # with our six oral-disease classes.
    model.classifier[-1] = nn.Linear(
        num_features,
        NUM_CLASSES
    )
 
    # Load our trained checkpoint.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
 
    # Handle both a raw state_dict and a
    # checkpoint dictionary.
    if isinstance(checkpoint, dict):
 
        if "model_state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )
 
        elif "state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["state_dict"]
            )
 
        else:
            model.load_state_dict(
                checkpoint
            )
 
    else:
        model.load_state_dict(
            checkpoint
        )
 
    # Move model to the selected device.
    model = model.to(DEVICE)
 
    # Switch to evaluation mode.
    model.eval()
 
    return model
 
 
# Load the trained model.
model = load_model()
 
 
# ============================================
# PREDICTION FUNCTION
# ============================================
 
def predict_image(image):
 
    # Make sure the uploaded image is RGB.
    image = image.convert("RGB")
 
    # Apply the exact same preprocessing used
    # during validation/testing.
    image_tensor = transform(image)
 
    # Add batch dimension:
    # [3, 224, 224] → [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0)
 
    # Move image to the same device as the model.
    image_tensor = image_tensor.to(DEVICE)
 
    # Disable gradient calculation because this
    # is inference rather than training.
    with torch.no_grad():
 
        # Generate model outputs.
        outputs = model(image_tensor)
 
        # Convert outputs into probabilities.
        probabilities = torch.softmax(
            outputs,
            dim=1
        )
 
        # Find the class with the highest probability.
        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )
 
    # Convert tensor values into normal Python values.
    predicted_index = predicted_index.item()
    confidence = confidence.item()
 
    # Return the predicted class and confidence.
    return (
        CLASS_NAMES[predicted_index],
        confidence,
        probabilities.squeeze().cpu()
    )
 
 
# ============================================
# STREAMLIT INTERFACE
# ============================================
 
# Application title.
st.title("🦷 Oral Disease Classifier")
 
# Short explanation.
st.write(
    "Upload an oral image and the trained "
    "EfficientNet-B0 model will classify it "
    "into one of six oral disease categories."
)
 
 
# ============================================
# MULTIPLE IMAGE UPLOADER
# ============================================
 
# Allow the user to upload multiple images
# at the same time.
uploaded_files = st.file_uploader(
    "Upload oral images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)
 
 
# ============================================
# PROCESS UPLOADED IMAGES
# ============================================
 
if uploaded_files:
 
    for uploaded_file in uploaded_files:
 
        # Open each uploaded image.
        image = Image.open(uploaded_file)
 
        # Display the uploaded image.
        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True
        )
 
        # Add prediction button (unique key per file).
        if st.button(
            f"🔍 Classify {uploaded_file.name}",
            key=uploaded_file.name
        ):
 
            predicted_class, confidence, probabilities = (
                predict_image(image)
            )
 
            # ========================================
            # DISPLAY MAIN RESULT
            # ========================================
 
            st.success(
                f"Prediction: {predicted_class}"
            )
 
            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )
 
            # ========================================
            # DISPLAY ALL CLASS PROBABILITIES
            # ========================================
 
            st.subheader(
                "Class Probabilities"
            )
 
            # Display the probability for every class.
            for index, class_name in enumerate(
                CLASS_NAMES
            ):
 
                probability = (
                    probabilities[index].item()
                )
 
                st.write(
                    f"**{class_name}**: "
                    f"{probability * 100:.2f}%"
                )
 
                # Visual probability bar.
                st.progress(
                    probability
                )
 
        st.divider()
 
 
# ============================================
# DEVICE INFORMATION
# ============================================
 
# Display which device is being used.
st.sidebar.write(
    f"**Inference device:** `{DEVICE}`"
)