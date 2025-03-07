import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the test dataset path
test_dir = r'C:\xampp\htdocs\skin_cancer_app\dataset\test'  # Update to your actual test dataset folder

# Data preprocessing for test data
test_datagen = ImageDataGenerator(rescale=1.0/255)  # Normalize pixel values

# Create test data generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),  # Same as training size
    batch_size=32,
    class_mode='categorical',  # Multi-class classification (4 categories)
    shuffle=False  # No need to shuffle test data
)

# Load the trained model
model_path = os.path.join(r'C:\xampp\htdocs\skin_cancer_app', 'best_model.keras')
model = keras.models.load_model(model_path)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n🔹 Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"🔹 Test Loss: {test_loss:.4f}")
