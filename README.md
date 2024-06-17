# Food Recognition and Calorie Estimation Model

This repository contains a deep learning model that recognizes food items from images and estimates their calorie content. The model enables users to track their dietary intake and make informed food choices.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)


## Introduction

The model is built using Convolutional Neural Networks (CNN) with TensorFlow and Keras. It consists of a shared convolutional base for feature extraction and two output layers: one for food classification and another for calorie estimation.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/food-calorie-estimation.git
    cd food-calorie-estimation
    ```

2. Install the required dependencies:
    ```bash
    pip install tensorflow keras numpy opencv-python pandas matplotlib
    ```

## Dataset

The dataset should be organized with images in a directory and labels in a CSV file. The CSV file (`labels.csv`) should have the following columns: `image`, `category`, and `calories`.


## Usage

1. **Train the Model**:
    ```bash
    python train.py
    ```

2. **Predict with the Model**:
    ```python
    import cv2
    import numpy as np
    from tensorflow.keras.models import load_model

    # Load the trained model
    model = load_model('food_recognition_calorie_estimation_model.h5')

    # Preprocess the new image
    img = cv2.imread('path_to_new_image.jpg')
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make a prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])
    estimated_calories = prediction[1][0][0]
    print(f'Predicted Class: {predicted_class}, Estimated Calories: {estimated_calories:.2f} kcal')
    ```

## Results

The model achieves good accuracy for food classification and reasonably accurate calorie estimation. Training and validation accuracy and loss plots are generated to visualize the model's performance.

