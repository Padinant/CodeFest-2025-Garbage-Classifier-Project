# CodeFest-2025-Garbage-Classifier-Project

# Oscar The G.A.I.A Green Automated Intelligent Assistant Garbage Detector

## Overview ##
This project implements a garbage classification model using MobileNet for feature extraction and Keras for training and inference. The model is designed to classify different types of waste into categories such as battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, and trash.

## Features ##
- Pretrained MobileNet Model: Utilizes MobileNet with `imagenet` weights for feature extraction.
- Custom Dataset Handling: Loads and preprocesses images from the dataset.
- Model Training: Fine-tunes MobileNet on a custom garbage dataset.
- Inference Support: Predicts waste category for new images.

## Dependencies
Ensure you have the following libraries installed before running the notebook:
```bash
pip install tensorflow keras numpy matplotlib
```

## Dataset
The dataset is structured in the following way:
```
/content/garbage-dataset/
    ├── battery/
    │   ├── battery_101.jpg
    │   ├── battery_102.jpg
    ├── biological/
    │   ├── biological_101.jpg
    ├── cardboard/
    │   ├── cardboard_101.jpg
    ...
```
Update `dataset_path` accordingly in the notebook if using a different dataset location.

## Step-by-Step Guide to Running the Program

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-repo/MobileNetGarbageDetector.git
cd MobileNetGarbageDetector
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset
- Ensure the dataset is placed in the correct directory (`/content/garbage-dataset/`).
- If using a custom dataset, modify the dataset path in the script accordingly.

### Step 4: Train the Model
Run the following command inside a Jupyter Notebook or Python script:
```python
python train.py
```
This will start training the MobileNet-based garbage classification model.

### Step 5: Run Inference on a Sample Image
To classify a new image, use the following script:
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model("MobileNetGarbageDetector.h5")

# Load and preprocess an image
img = image.load_img("path_to_image.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
print("Predicted class:", predicted_class)
```

### Step 6: Save and Load the Model
To save the trained model:
```python
model.save("MobileNetGarbageDetector.h5")
```
To load the saved model:
```python
from tensorflow.keras.models import load_model
model = load_model("MobileNetGarbageDetector.h5")
```

## Notes
- Ensure the dataset path is correctly set before running the notebook.
- Use GPU acceleration if available for faster training (`!nvidia-smi` to check GPU availability in Colab).
- Adjust `epochs` and `batch_size` in the training process for better performance.

## License
This project is open-source and available under the **MIT License**.


# Oscar The G.I.A.I Garbage in Artificial Intelligence out: Hybrid Garbage Classification Model (Still in Development)

## Overview
Oscar is a hybrid garbage classification model that combines Bayesian inference from PyMC with deep learning using TensorFlow and PyTorch. This model is designed to classify garbage images into various categories for improved recycling and waste management. The dataset is sourced from Kaggle's "Garbage Classification v2" dataset. .  

## Merging the Models
Oscar integrates:
1. A **Bayesian Neural Network (BNN)** using PyMC to estimate prediction uncertainties.
2. A **Sequential Deep Learning Model** with Depthwise Separable Convolutions (similar to MobileNet) for efficient image classification.

This fusion allows the model to leverage the interpretability and uncertainty estimation of Bayesian inference while maintaining the efficiency and accuracy of deep learning.

## Setup and Dependencies
Ensure you have the necessary dependencies installed:
```sh
pip install kaggle pymc pytensor torch torchvision tensorflow matplotlib seaborn numpy pandas
```

## Dataset Preparation
1. **Download the dataset** using Kaggle API.
2. **Process images:** Convert PNG to JPG, remove transparency, and resize images to 250x250 pixels.
3. **Split the dataset** into training (70%), validation (15%), and test (15%) sets.

## Model Architecture
### Bayesian Inference with PyMC
- Defines prior distributions for weights and biases.
- Uses **ADVI (Automatic Differentiation Variational Inference)** for posterior approximation.
- Provides uncertainty estimates for predictions.

### Deep Learning Model
- **Input:** Image of shape `(250, 250, 3)`.
- **Convolutional Layers:** Depthwise separable convolutions with Batch Normalization.
- **Pooling Layers:** MaxPooling for feature reduction.
- **Fully Connected Layers:** Dense layers with dropout to prevent overfitting.
- **Output Layer:** Softmax activation for 10-category classification.

## Training Process
The model is trained using **Sparse Categorical Crossentropy** with an Adam optimizer.
```python
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    metrics=['accuracy']
)
```
Training is performed in batches to optimize memory usage.

## Evaluation Metrics
- **Accuracy & Loss** (Train and Validation)
- **Precision, Recall, and F1 Score** (via classification report)
- **Confusion Matrix** (via Seaborn visualization)

## Model Saving
The trained model is saved in both **Keras** and **H5** formats for further training and deployment:
```python
tf.keras.models.save_model(model, '/content/my_models/Oscar_GarbageDetector.keras')
model.save('/content/my_models/Oscar_GarbageDetector.h5')
```

## Future Enhancements
- Implement data augmentation to improve generalization.
- Experiment with alternative Bayesian inference methods.
- Optimize inference speed using TensorFlow Lite or ONNX.

## Conclusion
Oscar provides an innovative hybrid approach to garbage classification by merging Bayesian inference with deep learning. This ensures both efficiency and uncertainty estimation, making it suitable for applications in waste management and environmental sustainability.




##########

Trash pymc model (Still in development)

###########

Bayesian ResNet Classification with PyMC and PyTorch


###########

Overview

###########


This project implements a Bayesian Neural Network (BNN) using PyMC v5 for probabilistic inference and PyTorch for feature extraction. The goal is to classify images of trash and recycling categories using a pre-trained ResNet50 model and Bayesian inference.


###########

Features

###########

Dataset Handling: Uses the Garbage Classification v2 dataset from Kaggle.

Pretrained Model: Employs a ResNet50 model to extract deep features.

Bayesian Inference: Implements a Bayesian Neural Network for classification using PyMC.

Probabilistic Predictions: Generates uncertainty estimates using Bayesian posterior samples.

###########
Installation
###########

Ensure you have the required dependencies installed:

pip install pymc pytensor torch torchvision kagglehub numpy

###########
Dataset
###########

The dataset is downloaded from KaggleHub:

sumn2u_garbage_classification_v2_path = kagglehub.dataset_download('sumn2u/garbage-classification-v2')
DATASET_PATH = sumn2u_garbage_classification_v2_path + "/garbage-dataset"

Ensure you have Kaggle API credentials set up to download the dataset.

#############

Usage

#############

1. Load Dataset and Preprocess Images

train_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

2. Extract Features Using ResNet50

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = nn.Identity()  # Remove classification layer for feature extraction
resnet.to(device)
X_train, y_train = extract_features(resnet, train_loader)

3. Train Bayesian Neural Network

with pm.Model() as bayesian_resnet:
    W = pm.Normal("W", mu=0, sigma=1, shape=(num_features, len(CATEGORIES)))
    b = pm.Normal("b", mu=0, sigma=1, shape=(len(CATEGORIES),))
    logits = pt.dot(X_train, W) + b
    y_pred = pm.Categorical("y_pred", pm.math.softmax(logits), observed=y_train)
    approx = pm.fit(n=50000, method=pm.ADVI())
    trace = approx.sample(1000)

4. Make Predictions

new_image_features = X_train[:1]
prediction = predict_pymc(new_image_features, trace)
print("Predicted Class:", CATEGORIES[np.argmax(prediction)])

################

Notes

################

Ensure GPU acceleration is enabled for faster processing (torch.cuda.is_available()).

Use a subset of images for testing if experiencing memory issues.

Bayesian inference is computationally intensive—consider reducing iterations for quick testing.

#################

License

#################

This project is open-source under the MIT License.

