# CodeFest-2025-Garbage-Classifier-Project

##########
Trash pymc model
##########

###########
Bayesian ResNet Classification with PyMC and PyTorch
###########

###########
Overview
###########

###########
This project implements a Bayesian Neural Network (BNN) using PyMC v5 for probabilistic inference and PyTorch for feature extraction. The goal is to classify images of trash and recycling categories using a pre-trained ResNet50 model and Bayesian inference.
###########

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

