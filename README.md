# Project GAIA: Green Automated Intelligent Assistant MobileNet Garbage Detector

## Project Overview

This repository contains a convolutional neural network (CNN) based garbage classification system that utilizes a fine-tuned MobileNet model to classify images into 10 different waste categories. The project was developed for CodeFest 2025 under Challenge #1 by Team #10: Padina Nasiri Toussi, Timothy Winans.

This model helps facilitate better waste segregation and recycling efforts using computer vision.

---
For instructions on how to run the executable, please see the instruction details in our executable file

## Dataset
The dataset used for training this model is publicly available on Kaggle:
[Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2?resource=download)

### Categories:
1. Battery
2. Biological
3. Cardboard
4. Clothes
5. Glass
6. Metal
7. Paper
8. Plastic
9. Shoes
10. Trash

Each category contains labeled images for supervised training.

## Strengths of the Model
- An already energy efficient and low cost base model (MobileNet), fine-tuned on the recycling dataset
- Uses Depthwise Separable Convolutions for efficiency, and reduced number of parameters.
- Enhances interpretability while maintaining high classification accuracy.
- Focus on other metrics beyond just accuracy, to minimize contaminated and missed recyclings
- Could be deployed on low-power devices like Raspberry Pi or mobile phones
- The dataset is processed efficiently, using image compression and resizing techniques to limit storage and computational overhead.
- Batch processing and on-the-fly augmentation reduce the need for high memory and storage requirements.

---
For general GitHub users:
## Installation & Setup

### **1. Clone Repository**
```bash
git clone https://github.com/your-username/MobileNet-Garbage-Detector.git
cd MobileNet-Garbage-Detector
```

### **2. Open FinalMobileNetGarbageDetector.ipynb (Recommended on Google Colab)**
This notebook was used to train the final model

It will:
- Load and preprocess the dataset (resizing images to 224x224, normalizing)
- Load the MobileNet base model with ImageNet weights
- Fine-tune the model on the 10-class dataset
- Provide useful analysis and metrics on model performance
- Save the trained model

### **3. Setup Kaggle API for Dataset Download**
Sign up at [Kaggle](https://www.kaggle.com) and create an API token:
1. Go to `Account` → `Create New API Token`
2. Save `kaggle.json` to `~/.kaggle/` (Linux/Mac) or `%HOMEPATH%\.kaggle\` (Windows)

Then, follow the instructions found in FinalMobileNetGarbageDetector to setup Kaggle API using your Kaggle username and key

### **4. Train and Save the Model**
Follow along with the FinalMobileNetGarbageDetector notebook instructions to train the model, and save a copy of it on google colab

### **5. Upload trained model on ProjectExecutable.ipynb (Recommended on Google Colab)**
Follow along with the instructions in ProjectExecutable, to upload the trained model on the notebook session storage.
Then, you can run the notebook to get multiple types of classification reports for individual or multiple images, based on your needs.
Among other things, the notebook also has optional informative helper functions to help you gather and preprocess your input data.

### **An Example Output**:
```
Predicted Class: Plastic
Top Three Prediction Probabilities: [Plastic: 0.85, Glass: 0.06, Battery: 0.03]
```
---

## Future Areas of Improvement
- Implement further data augmentation (such as rotating images) to improve generalization.
- Experiment with alternative Computer Vision models.
- Optimize inference speed using TensorFlow Lite or ONNX.
- Leverage Bayesian Neural Networks (BNNs) for uncertainty estimation.
- Integrate with a BNN to create a hybrid garbage classification model combining Bayesian inference from PyMC with deep learning using TensorFlow and PyTorch through an ensemble approach.  (For example, see the antique_code folder for early versions of the BNN model, Oscar The G.I.A.I, which was developed separately but in parallel with GAIA)

---

## Contributors
- **Padina Nasiri Toussi**
- **Timothy Winans**

For contributions, submit a **Pull Request (PR)** following our guidelines.

---

## License
MIT License © 2025 MobileNet Garbage Detector Project
