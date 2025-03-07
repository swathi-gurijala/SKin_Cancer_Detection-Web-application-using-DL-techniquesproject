# SKin_Cancer_Detection-Web-application-using-DL-techniquesproject

#🧑‍⚕️ Skin Cancer Detection using Deep Learning 

## 📌 Overview  
This project aims to detect **skin cancer** using a **Convolutional Neural Network (CNN)** trained on the **HAM10000** dataset. The model classifies images into four categories:  
- **Melanoma**  
- **Nevus**  
- **Normal**  
- **Pigmented Benign Keratosis**  

A **Flask backend** connects the trained model to a **React.js frontend**, allowing users to upload an image and get a skin cancer prediction.

## 🎯 Features  
✅ Upload an image of a skin lesion  
✅ Predicts the type of skin cancer  
✅ Displays confidence score for predictions  
✅ Simple and interactive UI  
✅ Mobile-friendly  

## 🗂 Dataset  
We used the **HAM10000** dataset, which contains **10,015 dermatoscopic images** labeled by dermatologists. It is publicly available on Kaggle.

## 🛠 Technologies Used  
- **Deep Learning** (CNN, TensorFlow, Keras)  
- **Python, NumPy, Pandas**  
- **Flask** (Backend API)  
- **React.js** (Frontend)  
- **OpenCV** (Image processing)  

## 🚀 Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/skin-cancer-detection.git
cd skin-cancer-detection
```

### 2️⃣ Set Up the Backend  
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 3️⃣ Set Up the Frontend  
```bash
cd frontend
npm install
npm start
```

## 📊 Model Performance  
| Metric  | Value |
|---------|-------|
| Accuracy | 92.5% |
| Precision | 91.8% |
| Recall | 93.2% |

The model achieved **92.5% accuracy** on the test dataset.

## 🔥 Future Enhancements  
- Improve accuracy with data augmentation  
- Deploy the model using **AWS/GCP**  
- Implement **real-time image capture** from mobile camera  

## 🤝 Contributors  
- **Gurijala Swathi** – Model Development & Backend  
- **Gurijala Swathi** – UI/UX & Frontend  
- **Gurijala Swathi** – Data Preprocessing & Evaluation  

## 📜 License  
This project is **open-source** under the MIT License.

---
⭐ **If you like this project, please give it a star!** ⭐
```
