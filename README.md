# Project GAIA: Green Automated Intelligent Assistant

## Project Overview

This repository contains a deep-learning-based **garbage classification system** that utilizes a fine-tuned **MobileNet** model to classify images into 10 different waste categories. The project was developed for **CodeFest 2025** under Challenge #1 by Team #10: **Padina Nasiri Toussi, Timothy Winans**.

This model helps facilitate better waste segregation and recycling efforts using computer vision.

---

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

---

## Installation & Setup

### **1. Clone Repository**
```bash
git clone https://github.com/your-username/MobileNet-Garbage-Detector.git
cd MobileNet-Garbage-Detector
```

### **2. Setup Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\\Scripts\\activate'
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Setup Kaggle API for Dataset Download**
Sign up at [Kaggle](https://www.kaggle.com) and create an API token:
1. Go to `Account` → `Create New API Token`
2. Save `kaggle.json` to `~/.kaggle/` (Linux/Mac) or `%HOMEPATH%\.kaggle\` (Windows)

Then run:
```bash
pip install kaggle
kaggle datasets download -d sumn2u/garbage-classification-v2
unzip garbage-classification-v2.zip -d data
```

---

## Training the Model

Run the training script to fine-tune **MobileNet** on the garbage dataset:
```bash
python train.py
```
This will:
- Preprocess the dataset (resizing images to 224x224, normalizing)
- Load the MobileNet model with ImageNet weights
- Fine-tune the model on the 10-class dataset
- Save the best model checkpoint

### **Hyperparameters Used**:
- Optimizer: **Adam**
- Learning Rate: **0.001**
- Batch Size: **32**
- Epochs: **25**

To adjust parameters, edit `config.py`.

---

## Running Inference
Once trained, the model can classify new images.

### **1. Run the Inference Script**
```bash
python predict.py --image_path test_image.jpg
```

### **2. Example Output**:
```
Predicted Class: Plastic
Prediction Probabilities: [Battery: 0.1, Biological: 0.05, Cardboard: 0.02, ... Plastic: 0.85, Trash: 0.05]
```

---

## Model Performance

The trained model achieves **high accuracy** on test images.

| Metric          | Value |
|----------------|-------|
| Accuracy       | 91.3% |
| Precision      | 89.5% |
| Recall         | 90.2% |
| F1 Score       | 89.8% |

#### **Confusion Matrix**:
A confusion matrix is generated using:
```python
python evaluate.py
```

---

## Troubleshooting
**Common Issues & Fixes**
| Issue | Solution |
|--------|-----------|
| `ModuleNotFoundError: No module named 'tensorflow'` | Run `pip install tensorflow keras` |
| Kaggle API not found | Ensure `kaggle.json` is placed correctly |
| Out of memory (OOM) on GPU | Reduce batch size: `batch_size = 16` |

---

## Future Improvements
uture Enhancements
- Implement data augmentation to improve generalization.
- Experiment with alternative Bayesian inference methods.
- Optimize inference speed using TensorFlow Lite or ONNX.
- Integrate Oscar The G.I.A.I (Garbage in Artificial Intelligence out): A hybrid garbage classification model combining Bayesian inference from PyMC with deep learning using TensorFlow and PyTorch. 
  - Leverages Bayesian Neural Networks (BNNs) for uncertainty estimation.
  - Uses Depthwise Separable Convolutions similar to MobileNet for efficiency.
  - Enhances interpretability while maintaining high classification accuracy.
- Develop a Bayesian ResNet Classification Model for additional probabilistic classification capabilities.

---

## Contributors
- **Padina Nasiri Toussi**
- **Timothy Winans**

For contributions, submit a **Pull Request (PR)** following our guidelines.

---

## License
MIT License © 2025 MobileNet Garbage Detector Project
