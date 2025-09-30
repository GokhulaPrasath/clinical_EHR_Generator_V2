import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from scipy.ndimage import uniform_filter, generic_filter
import os
from PIL import Image
import streamlit as st

def build_srcnn_model():
    """Build SRCNN model for image enhancement"""
    input_img = Input(shape=(None, None, 1))
    x = Conv2D(64, (9, 9), padding='same', activation='relu')(input_img)
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    output = Conv2D(1, (5, 5), padding='same', activation='linear')(x)
    model = Model(input_img, output)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_classification_model(input_shape=(256, 256, 1)):
    """Build classification model for cardiac condition analysis"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_and_process_slices(folder_path, num_slices, target_size=(256, 256), enhance=True):
    """
    Load actual slices from folder path instead of generating synthetic ones
    """
    processed_slices = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        st.error(f"Folder path does not exist: {folder_path}")
        return processed_slices
    
    # Get list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    image_files.sort()  # Ensure consistent order
    
    # If the number of image files doesn't match num_slices, use what's available
    if len(image_files) != num_slices:
        st.warning(f"Number of images in folder ({len(image_files)}) doesn't match metadata num_slices ({num_slices}). Using available images.")
    
    # Load and process each image
    for i, image_file in enumerate(image_files[:num_slices]):
        try:
            img_path = os.path.join(folder_path, image_file)
            
            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                st.error(f"Failed to load image: {img_path}")
                continue
            
            # Resize to target size
            img = cv2.resize(img, target_size)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            processed_slices.append(img)
            
        except Exception as e:
            st.error(f"Error processing image {image_file}: {e}")
    
    return processed_slices

def enhance_with_srcnn(slices, srcnn_model):
    """Enhance slices using SRCNN model"""
    enhanced_slices = []
    for slice in slices:
        slice_expanded = np.expand_dims(slice, axis=0)
        slice_expanded = np.expand_dims(slice_expanded, axis=-1)
        enhanced = srcnn_model.predict(slice_expanded, verbose=0)
        enhanced = np.squeeze(enhanced)
        enhanced_slices.append(enhanced)
    return enhanced_slices

def calculate_entropy(image):
    """Calculate image entropy"""
    hist = np.histogram(image, bins=256, range=(0, 1))[0]
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy

def calculate_homogeneity(image):
    """Calculate image homogeneity"""
    std_dev = generic_filter(image, np.std, size=5)
    homogeneity = 1 / (1 + np.mean(std_dev))
    return homogeneity

def calculate_symmetry(image):
    """Calculate cardiac symmetry"""
    height, width = image.shape
    left_half = image[:, :width//2]
    right_half = image[:, width//2:]
    right_flipped = np.fliplr(right_half)
    if left_half.shape == right_flipped.shape:
        mse = np.mean((left_half - right_flipped) ** 2)
        symmetry = 1 / (1 + mse)
        return symmetry
    return 0.5

def estimate_cardiac_area(image):
    """Estimate cardiac area from image"""
    _, thresh = cv2.threshold((image * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        return area / (image.shape[0] * image.shape[1])
    return 0

def extract_image_features(slices):
    """Extract features from image slices"""
    features = []
    for slice in slices:
        slice_features = {
            'mean_intensity': np.mean(slice),
            'std_intensity': np.std(slice),
            'max_intensity': np.max(slice),
            'min_intensity': np.min(slice),
            'contrast': np.max(slice) - np.min(slice),
            'entropy': calculate_entropy(slice),
            'homogeneity': calculate_homogeneity(slice),
            'cardiac_area': estimate_cardiac_area(slice),
            'symmetry_score': calculate_symmetry(slice)
        }
        features.append(slice_features)
    return features