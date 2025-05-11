import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.metrics import accuracy_score, confusion_matrix

# Data Preprocessing
def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)

def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    return image

# Augmentation
def augment_image(image):
    if np.random.rand() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(np.random.uniform(0.5, 1.5))
    return image

# Model Utilities
def load_model(model_path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    return model

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

# Evaluation Metrics
def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

# File Handling
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_file_paths(directory, extension=".png"):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]

# Visualization
import matplotlib.pyplot as plt

def plot_loss_curve(loss_history):
    plt.plot(loss_history)
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def visualize_predictions(images, predictions, ground_truths):
    fig, axs = plt.subplots(1, len(images))
    for ax, img, pred, truth in zip(axs, images, predictions, ground_truths):
        ax.imshow(img)
        ax.set_title(f"Pred: {pred}, Truth: {truth}")
        ax.axis('off')
    plt.show()
