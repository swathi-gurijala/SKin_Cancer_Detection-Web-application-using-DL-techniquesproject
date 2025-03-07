from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)  # Initialize Flask app

# Enable CORS for all routes
CORS(app)

# Load trained model
MODEL_PATH = "skin_cancer_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Define class labels
class_labels = ["Melanoma", "Nevus", "Normal", "Pigmented Benign Keratosis"]

# Define risk levels and descriptions based on class
risk_levels = {
    "Normal": {
        "risk": "No risk",
        "description": "No signs of skin cancer. Healthy skin, but regular checkups are recommended to maintain overall skin health."
    },
    "Nevus": {
        "risk": "No risk",
        "description": "A benign mole or birthmark. It's generally harmless, but keep an eye out for any changes in size, shape, or color."
    },
    "Pigmented Benign Keratosis": {
        "risk": "Low risk",
        "description": "A benign (non-cancerous) skin condition, often appearing as a pigmented patch. While usually harmless, it may require monitoring or treatment if it becomes irritated."
    },
    "Melanoma": {
        "risk": "High risk",
        "description": "A malignant (cancerous) skin condition that can spread to other parts of the body. Immediate consultation with a dermatologist is crucial."
    }
}

@app.route("/", methods=["GET"])
def home():
    return "✅ Skin Cancer Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_path = "uploaded_image.jpg"
    file.save(img_path)

    # Preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    
    # Get risk level and description based on prediction
    risk = risk_levels[predicted_class]["risk"]
    description = risk_levels[predicted_class]["description"]

    return jsonify({
        "prediction": predicted_class,
        "risk": risk,
        "description": description
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Ensure Flask listens on all IPs
