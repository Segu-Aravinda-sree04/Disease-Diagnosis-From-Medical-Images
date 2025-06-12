# 🧬 Disease Diagnosis from Medical Images using CNNs

This project implements a deep learning-based diagnostic tool that classifies histopathological images of lung and colon tissue into different disease categories using Convolutional Neural Networks (CNNs). Built using TensorFlow and Keras, it aims to support early diagnosis in clinical settings through accurate image-based classification.

---

## 🧠 Overview

- 📂 Dataset: [Lung and Colon Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- 🏷️ Classes: Lung Adenocarcinoma, Lung Benign Tissue, Lung Squamous Cell Carcinoma, Colon Adenocarcinoma, Colon Benign Tissue
- ⚙️ Model: Custom deep CNN with multiple Conv2D layers
- 📊 Evaluation: Train / Validation / Test splits with accuracy & loss metrics

---

## ✅ Features

| Feature                     | Description                                                  |
|----------------------------|--------------------------------------------------------------|
| 🖼️ Image Classification     | Diagnoses diseases from histopathological images             |
| 📊 Model Performance        | Accuracy, loss, and confusion matrix on test dataset         |
| 🧠 CNN Architecture         | 5 convolutional blocks, each with increasing filters         |
| ⚙️ Optimizer                | Adamax with categorical crossentropy                         |
| 🗃️ Data Handling            | Uses `flow_from_dataframe()` with data generators            |
| 📈 Evaluation Reports       | Generates predictions and classification reports             |

---

## 🔮 Future Enhancements

- 🩺 Support for more diseases and datasets

- 🧠 Transfer learning with ResNet/EfficientNet

- 📈 Grad-CAM heatmaps for model interpretability

- 🌐 Web interface for diagnosis upload

