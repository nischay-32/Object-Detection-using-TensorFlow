# Object Detection: Digit Classification and Localization with TensorFlow

This repository contains a Python implementation of an object detection model built with TensorFlow and Keras. The project demonstrates how to build a Convolutional Neural Network (CNN) that can simultaneously perform **classification** (identifying a digit) and **localization** (predicting its bounding box).

The entire project is contained within a single Google Colab notebook, making it easy to run and reproduce.

![Model Predictions on Validation Set](https://storage.googleapis.com/grounded-video-pipeline-prod/dc01d2d057a74a16b9b3240e5ebf860f/dc01d2d057a74a16b9b3240e5ebf860f_0.jpeg)
*(Image: Sample output showing the model's predictions on the validation set. Green boxes indicate correct classifications, while red boxes indicate incorrect ones.)*

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Methodology](#-methodology)
  - [Synthetic Data Generation](#1-synthetic-data-generation)
  - [Model Architecture](#2-model-architecture)
  - [Training and Evaluation](#3-training-and-evaluation)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Technologies Used](#-technologies-used)

---

## 📌 Project Overview

Object detection is a fundamental task in computer vision. This project tackles the two core components of object detection:

1.  **Classification**: Determining *what* object is in an image (e.g., a digit '7').
2.  **Localization**: Determining *where* the object is in the image via bounding box coordinates.

To simplify the problem and focus on the model architecture, this project uses a synthetically generated dataset. Handwritten digits from the MNIST dataset are randomly placed onto larger blank canvases, with their bounding box coordinates calculated automatically. This provides a clean, perfectly labeled dataset for training.

---

## ⚙️ Methodology

### 1. Synthetic Data Generation

A custom dataset was created using the MNIST handwritten digit database.
- A 75x75 pixel blank canvas was prepared.
- A 28x28 MNIST digit was randomly placed on the canvas.
- The bounding box coordinates (`x_min`, `y_min`, `x_max`, `y_max`) were calculated and normalized.
- This process was implemented using a `tf.data.Dataset` generator for efficient data pipeline management.

### 2. Model Architecture

A custom CNN was built with a shared feature extraction base and two distinct output heads.

- **Data Augmentation**: The model begins with `tf.keras.layers` for random rotation, zoom, translation, and contrast adjustments to make the model more robust and prevent overfitting.
- **Feature Extractor**: A stack of three `Conv2D` and `MaxPooling2D` layers learns spatial features from the input images.
- **Dual Output Heads**: The network splits to handle the two distinct tasks:
  - **Classification Head**: A `Dense` layer with a **Softmax** activation predicts the digit class (0-9).
  - **Localization Head**: A `Dense` layer with a linear activation function regresses the four bounding box coordinates.

![Model Architecture Diagram](https://storage.googleapis.com/grounded-video-pipeline-prod/dc01d2d057a74a16b9b3240e5ebf860f/dc01d2d057a74a16b9b3240e5ebf860f_1.jpeg)
*(Image: A simplified diagram of a multi-output model architecture.)*


### 3. Training and Evaluation

- **Loss Functions**: The model was compiled with two loss functions: `sparse_categorical_crossentropy` for classification and `mean_squared_error` (MSE) for the bounding box regression.
- **Optimizer**: The **Adam** optimizer was used with a fine-tuned learning rate of `0.0001` for stable convergence.
- **Training**: The model was trained for 30 epochs, reaching a final validation classification accuracy of over **97%**.

---

## ▶️ How to Run

This project is designed to be run in a Google Colab environment.

1.  **Clone the repository or download the `.ipynb` file.**
2.  **Open the notebook in Google Colab.**
    - Go to [colab.research.google.com](https://colab.research.google.com).
    - Click on `File -> Upload notebook` and select the downloaded `.ipynb` file.
3.  **Run the cells.**
    - Click on `Runtime -> Run all`.
    - The notebook will automatically install the required dependencies, generate the dataset, build the model, train it, and display the final evaluation results.

---

## 📊 Results

The model performs exceptionally well on the synthetic validation data.

- **Classification Accuracy**: Achieved **~97.6%** accuracy on the validation set.
- **Localization**: The predicted bounding boxes closely match the ground truth locations of the digits. Visual inspection shows that even when the classification is incorrect, the localization is often accurate.

The training history plots for accuracy and loss are shown below:
![Training History Plots](https://storage.googleapis.com/grounded-video-pipeline-prod/dc01d2d057a74a16b9b3240e5ebf860f/dc01d2d057a74a16b9b3240e5ebf860f_2.jpeg)

---

## 🛠️ Technologies Used

- **TensorFlow & Keras**: For building and training the deep learning model.
- **NumPy**: For numerical operations and data manipulation.
- **Matplotlib**: For plotting and visualizing results.
- **Pillow (PIL)**: For image manipulation and drawing bounding boxes.
- **Google Colab**: As the development and runtime environment.
