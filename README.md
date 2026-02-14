# ML Assignment – Supervised Regression
## Predicting Pixel Coordinates in a 50×50 Grayscale Image

---

## 📌 Problem Statement

Using Deep Learning techniques, predict the coordinates **(x, y)** of a pixel which has a value of **255** in a given **50×50 grayscale image**, where all other pixels have a value of **0**.

This problem is formulated as a **supervised regression task**, where the model predicts two continuous values representing the pixel location.

---

## 🧠 Approach

### Problem Formulation

- Input: 50×50 grayscale image
- Output: Two continuous values → (x, y) coordinates
- Learning Type: Supervised Learning
- Task Type: Multi-output Regression

Since the output consists of continuous coordinate values, this is treated as a regression problem instead of classification.

---

## 📊 Dataset Design & Rationale

Since dataset generation was permitted, a synthetic dataset was created.

### Dataset Characteristics

- Image size: 50 × 50
- Pixel values:
  - One pixel = 255
  - All others = 0
- Coordinates randomly sampled from:
  - x ∈ [0, 49]
  - y ∈ [0, 49]

### Rationale Behind Dataset Choices

- Synthetic generation ensures perfect ground truth labels.
- Uniform distribution guarantees balanced coordinate coverage.
- Eliminates external bias or noise.
- Fully reproducible.

---

## ⚙️ Preprocessing

- Images normalized to range [0, 1]
- Optional coordinate normalization for stable training
- Train-validation split used

---

## 🏗 Model Architecture

A Convolutional Neural Network (CNN) was used due to its ability to learn spatial relationships.

### Architecture Overview

- Conv2D + ReLU
- MaxPooling
- Conv2D + ReLU
- MaxPooling
- Flatten
- Dense + ReLU
- Dense (2 units → x and y)

---

## 🔧 Training Configuration

- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)
- Batch Size: Configurable
- Epochs: Configurable

### Why MSE?

MSE penalizes large prediction errors more strongly and provides smooth gradients for optimization, making it suitable for regression tasks.

---

## 📈 Evaluation

The notebook includes:

- Training loss curves
- Validation loss curves
- Visualization of ground truth vs predicted coordinates
- Error distribution analysis

The model successfully learns to predict pixel coordinates with low regression error.

---

## 📦 Installation

### Requirements

- Python 3.9+
- TensorFlow
- NumPy
- Matplotlib
- Jupyter Notebook

### Install Dependencies

```bash
pip install tensorflow numpy matplotlib jupyter
