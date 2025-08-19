🐶🐱 Dog vs Cat Classification using Transfer Learning

This project demonstrates image classification using transfer learning to distinguish between dogs and cats. The notebook covers dataset extraction, preprocessing, model training, evaluation, and predictions using deep learning techniques.

📂 Project Structure
📦 Dog_vs_Cat_Classification
 ┣ 📜 DL_Project_3_Dog_vs_Cat_Classification_Transfer_Learning.ipynb  # Main Jupyter Notebook
 ┣ 📜 kaggle.json                                                     # Kaggle API key (not uploaded in repo)
 ┣ 📜 README.md                                                       # Documentation
 ┣ 📜 requirements.txt                                                # Python dependencies

⚙️ Installation & Dependencies

Install dependencies:

cd Dog-vs-Cat-Classification
pip install -r requirements.txt

Required Libraries

kaggle – to download dataset from Kaggle competitions

numpy & pandas – data handling

matplotlib & seaborn – visualizations

tensorflow / keras – deep learning models

os, zipfile – dataset extraction & file handling

📊 Dataset

Dataset: Dogs vs Cats (Kaggle Competition)

Images are divided into training and validation sets.

Preprocessing includes resizing, normalization, and data augmentation.

🚀 Workflow

Dataset Download & Extraction

Download dataset using Kaggle API

Extract .zip files and organize into train/test directories

Data Preprocessing

Image resizing & normalization

Data augmentation for better generalization

Model Building (Transfer Learning)

Use pre-trained models like VGG16, ResNet, or Inception

Freeze base layers and add custom dense layers for classification

Training & Validation

Train on augmented dataset

Monitor performance using accuracy and loss curves

Evaluation

Evaluate model on validation/test dataset

Generate accuracy, confusion matrix, and classification report

Prediction

Predict if a new image is a Dog 🐶 or a Cat 🐱

📌 Example Prediction
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess image
img = image.load_img("sample.jpg", target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Prediction
prediction = model.predict(img_array)
print("Prediction:", "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱")
