Brain Tumor Detection 🧠




Project Overview

This project focuses on automated brain tumor detection from MRI images using deep learning. It uses transfer learning with the EfficientNetB3 model and is designed to classify brain MRI scans accurately into tumor categories.

Key Goals:

Train a CNN-based model for tumor detection

Visualize model performance

Provide a ready-to-use saved model (Model.h5) for inference

Features

✅ Download and preprocess brain MRI dataset from Kaggle

✅ Data augmentation and normalization using ImageDataGenerator

✅ Deep learning model using TensorFlow and Keras

✅ Transfer learning with EfficientNetB3

✅ Training, validation, and testing pipelines

✅ Visualize training history (loss & accuracy)

✅ Confusion matrix & classification report

✅ Save and reuse trained model

Dataset

Dataset: Brain MRI Images for Brain Tumor Detection
 by Navoneel on Kaggle.

Classes include multiple tumor types

MRI scans are provided in RGB format

Installation

Clone the repository

git clone <your-repo-url>
cd brain-tumor-detection


Set up virtual environment (optional)

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate


Install dependencies

pip install -r requirements.txt


Dependencies include:

Python 3.9+

TensorFlow, Keras

scikit-learn

OpenCV (cv2)

numpy, pandas, matplotlib, seaborn

PIL, KaggleHub

Usage

Download dataset

import kagglehub
kagglehub.dataset_download('navoneel/brain-mri-images-for-brain-tumor-detection')


Run the main script

python brain_tumor_detection.py


Training

Default: 30 epochs

Visualizes training & validation accuracy and loss

Evaluation

Confusion matrix

Classification report

Model saved as Model.h5

Model Architecture

Base Model: EfficientNetB3 (pre-trained on ImageNet)

Added Layers:

BatchNormalization

Dense (with L1 & L2 regularization)

Dropout (0.45)

Output Dense layer with softmax activation

Optimizer: Adamax

Loss Function: Categorical Crossentropy

Results

Training Accuracy & Loss

Validation Accuracy & Loss

Confusion Matrix

Sample Plots:
(Placeholders for GitHub images)






Saving & Loading Model

Save model after training:

model.save('Model.h5')


Load model for inference:

from tensorflow.keras.models import load_model
model = load_model('Model.h5')

Acknowledgements

Dataset by Navoneel on Kaggle

TensorFlow & Keras for deep learning framework

EfficientNet architecture for transfer learning
