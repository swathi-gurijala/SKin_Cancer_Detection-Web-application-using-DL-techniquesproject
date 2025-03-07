import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

# Example model definition
def create_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Adjust based on your problem
    return model

# Example data (replace this with your actual training data)
# Make sure to use your real data and labels here
input_shape = (64, 64, 3)  # Example input shape for images (adjust as needed)
X_train = np.random.rand(100, *input_shape)  # Random data for demonstration
y_train = np.random.randint(0, 2, size=(100,))  # Random binary labels for demonstration

# Create and compile the model
model = create_model(input_shape)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (adjust epochs and batch size as necessary)
model.fit(X_train, y_train, epochs=5, batch_size=10)

# Check if the directory exists
save_path = r'C:\xampp\htdocs\skin_cancer_app'
if not os.path.exists(save_path):
    print("Directory does not exist.")
else:
    # Save the model
    model.save(os.path.join(save_path, 'model.h5'))
    print("Model saved successfully.")
