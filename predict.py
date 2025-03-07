import cv2
import keras
import numpy as np
import sys

# Define class labels (same order as in training)
class_labels = ["melanoma", "nevus", "normal", "pigmented benign keratosis"]

# Load the trained model
model_path = r"C:\xampp\htdocs\skin_cancer_app\best_model.keras"
model = keras.models.load_model(model_path)

# Open webcam for real-time capture
cap = cv2.VideoCapture(0)  # 0 for default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

print("Press 'SPACE' to capture an image and analyze it.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Display the live video feed
    cv2.imshow("Live Skin Cancer Detection", frame)

    # Press 'SPACE' to capture and process image
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # ASCII for SPACE key
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()

# Process the captured frame
frame = cv2.resize(frame, (150, 150))  # Resize to model input size
frame = frame / 255.0  # Normalize
frame = np.expand_dims(frame, axis=0)  # Add batch dimension

# Make prediction
prediction = model.predict(frame)
predicted_index = np.argmax(prediction)  # Get class index
predicted_class = class_labels[predicted_index]  # Map to class label
confidence = np.max(prediction)  # Get highest confidence value

# Display result
print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")

# Provide medical advice based on predicted class
if predicted_class == "normal":
    print("Advice: No risk detected. Maintain regular skin check-ups.")
elif predicted_class == "nevus":
    print("Advice: Benign mole detected. Monitor for any changes.")
elif predicted_class == "pigmented benign keratosis":
    print("Advice: Benign skin condition. Consult a dermatologist if irritation occurs.")
elif predicted_class == "melanoma":
    print("Advice: High risk! Immediate consultation with a dermatologist is recommended.")
