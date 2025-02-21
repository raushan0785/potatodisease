# 🌿 Potato Leaf Disease Classification System 🍃

## 🚀 Overview

This project is a Deep Learning-based Leaf Disease Classification System designed to detect and classify diseases in potato plants. Using Convolutional Neural Networks (CNNs), this system ensures fast and accurate disease diagnoses, assisting farmers and agricultural researchers in identifying plant health issues early.

## ✨ Features

✅ Automated Disease Detection – Classifies different potato leaf diseases with high accuracy.✅ User-Friendly Interface – Deployable as a web or mobile application.✅ Fast & Efficient – Provides real-time predictions for early intervention.✅ High Accuracy – Trained on a dataset of healthy and diseased potato leaves.

## 📊 Dataset

The model is trained on a dataset containing images of healthy and diseased potato leaves, covering common diseases such as:
🌱 Early Blight🍂 Late Blight🌿 Healthy Leaves

## 🔧 Technologies Used

🧠 Deep Learning: TensorFlow / PyTorch

🐍 Programming Language: Python

## 🛋 Libraries: OpenCV, NumPy, Pandas, Matplotlib

🏢 Model Architecture: Convolutional Neural Networks (CNN)

🌐 Deployment Options:  Streamlit (for web applications)

## 🛠 Installation & Setup

## 1️⃣ Clone the repository:

 git clone https://github.com/your-username/potatodisease.git
 cd potato-leaf-disease-classification

## 2️⃣ Install dependencies:

 pip install -r requirements.txt

## 3️⃣ Download the dataset and place it in the appropriate directory.

🎯 Usage

🔹 Train the model:

 python train.py

🔹 Test the model:

 python test.py

🔹 Run inference on an image:

 python predict.py --image path/to/image.jpg

🔹 Deploy the model as a web app:

 python app.py

📊 Model Training

🔹 Dataset split into training, validation, and testing sets.

🔹 CNN model trained using Adam optimizer and Cross-Entropy Loss.

🔹 Augmentation techniques like rotation, flipping, and contrast adjustment applied.

📈 Quick Prediction Code Example

Here’s a simple Python script to load a trained model and classify a potato leaf image:

import tensorflow as tf
import numpy as np
import cv2
import sys

# Load trained model
model = tf.keras.models.load_model("potato_leaf_disease_model.h5")

# Define categories
CATEGORIES = ["Healthy", "Early Blight", "Late Blight"]

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Reshape for model input
    return img

# Load and predict
image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_leaf.jpg"
image = preprocess_image(image_path)
prediction = model.predict(image)
print(f"Predicted Class: {CATEGORIES[np.argmax(prediction)]}")

Run the script with:

python predict.py path/to/leaf_image.jpg

🎯 Results

🌟 Achieved high accuracy (>90%) on test data.🌟 Capable of classifying leaf images into healthy, early blight, and late blight categories.

🔮 Future Enhancements

🚀 Expand dataset to include more plant species.📱 Implement a mobile app for real-time disease detection.⚡ Improve model efficiency for faster predictions.

🤝 Contributors

👨‍💻 Your Name - 

🐟 License

📄 This project is licensed under the MIT License.

📚 References

📌 PlantVillage Dataset📌 Research papers on CNN-based plant disease detection

For any issues, feel free to raise an issue or submit a pull request. 🚀
