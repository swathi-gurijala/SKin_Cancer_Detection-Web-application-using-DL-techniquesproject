import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

# Define dataset path
train_dir = "dataset/train"

# Define hyperparameters
IMG_SIZE = 224  
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Image data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Load training dataset
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Get number of classes
num_classes = len(train_generator.class_indices)

# Define the MobileNetV2 base model
input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_tensor)

# Add custom layers
x = base_model.output
x = Flatten()(x)  
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=outputs)

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS
)

# Save the trained model
model.save("skin_cancer_model.h5")

print("✅ Training complete! Model saved as 'skin_cancer_model.h5'.")
