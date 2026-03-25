import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
from random import SystemRandom

IMG_SIZE = 224
DATASET_PATH = "dataset/"

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def load_data():
    X = []
    y = []
    
    for label, category in enumerate(classes):
        path = os.path.join(DATASET_PATH, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            X.append(image)
            y.append(label)
    
    return np.array(X)/255.0, np.array(y)

X, y = load_data()

# CNN Model (Feature Extractor)
cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu')  # Feature layer
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn.fit(X, y, epochs=5)

# Extract Features
features = cnn.predict(X)

# Save features
np.save("features.npy", features)
np.save("labels.npy", y)

# Save CNN model
cnn.save("cnn_model.h5")