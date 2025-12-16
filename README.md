# 🌸 Flower Image Classification with Transfer Learning

This project demonstrates an **image classification pipeline using Transfer Learning** with **MobileNetV2** and **TensorFlow Hub**. The objective is to classify flower images into five categories using a pre-trained convolutional neural network as a feature extractor.

The implementation highlights how transfer learning can significantly reduce training time and computational cost while maintaining strong performance on image classification tasks.

---

## 📌 Project Overview

- Uses **MobileNetV2** pre-trained on ImageNet as a feature extractor
- Applies **Transfer Learning** by freezing the convolutional backbone
- Trains a custom Dense classification head
- Performs multi-class image classification with **5 flower categories**:
  - Roses  
  - Daisies  
  - Dandelions  
  - Sunflowers  
  - Tulips  

---

## 🧠 Model Architecture

- **Backbone:** MobileNetV2 (TensorFlow Hub – feature vector)
- **Input shape:** 224 × 224 × 3
- **Trainable layers:** Final Dense layer only
- **Loss function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy

---

## 🗂 Dataset

- Public **TensorFlow Flowers Dataset**
- Automatically downloaded and extracted
- Images are resized to 224×224
- Pixel values normalized to the range [0, 1]
- Dataset split into training and testing sets using `train_test_split`

---

## ⚙️ Technologies & Libraries

- Python
- TensorFlow & Keras
- TensorFlow Hub
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

---

## 🚀 Key Learning Objectives

- Understand Transfer Learning with Convolutional Neural Networks
- Use pre-trained models from TensorFlow Hub
- Build and train a custom image classifier
- Perform image preprocessing and dataset preparation
- Evaluate model performance on unseen data

---

## 📈 Results

The model is trained for a few epochs and evaluated on a test set, achieving solid accuracy given the lightweight architecture and short training time. This approach demonstrates an efficient and scalable solution for image classification problems.

---

## 📄 Notes

This project is intended for educational purposes and serves as a practical example of how to apply transfer learning in computer vision tasks using TensorFlow.

