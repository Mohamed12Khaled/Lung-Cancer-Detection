# Lung Cancer Detection using Deep Learning (MSNN Enhancement Project)

##  Project Overview

This project aims to replicate and enhance the hybrid deep learning model "Maximum Sensitivity Neural Network (MSNN)" described in the research paper:


**Title:** Hybrid Deep Convolution Model for Lung Cancer Detection with Transfer Learning

**Authors:** \[Sugandha Saxena, S. N. Prasad, Ashwin M Polnaya, Shweta Agarwala]

**Institution:** 1 School of Electronics and Communication Engineering, REVA University Bangalore, India



2 Department of Electronics and Communication Engineering, Manipal Institute of Technology Bengaluru, 
Manipal Academy of Higher Education, Manipal. Bengaluru, India



3 Department of Radiology, Interventional Radiology Division, A J Hospital & Research Centre, Kuntikana, 
Mangalore, India



4 Department of Electrical and Computer Engineering, Aarhus University, Aarhus 8200, Denmark

**Dataset:** Chest CT scan images (249 cancerous, 185 normal)

**License:** CC BY 4.0 \[[Sugandha Saxena, S. N. Prasad, Ashwin M Polnaya, Shweta Agarwala](https://creativecommons.org/licenses/by/4.0/)]

The model utilizes a hybrid architecture: a 5-block Convolutional Neural Network (CNN) based on AlexNet for feature extraction and a K-Nearest Neighbor (KNN) classifier for classification. The original paper achieved up to 98% accuracy and 97% sensitivity.

---

##  Objective of the Project
This project is conducted as part of a master's research program. The core purpose is to apply advanced and modern deep learning techniques in healthcare, specifically targeting the diagnosis of lung cancer from CT scans. By addressing an existing clinical challenge—early and accurate detection of lung cancer—the project aims to bridge the gap between AI research and practical clinical utility.


* Reproduce the MSNN model on   public datasets.
* Improve model generalizability and explainability.
* Enhance clinical interpretability for radiologists.
* Deploy a prototype tool for medical use.

---

##  Proposed Enhancements

### 1. Model Improvements

* Replace AlexNet with a modern backbone (e.g., ResNet50, EfficientNet, DenseNet).
* Use advanced pooling techniques (e.g., Global Average Pooling + Attention).
* Replace KNN with a fully-connected classifier for better end-to-end training.

### 2. Evaluation Enhancements

* Add metrics: ROC Curve, AUC, Specificity, Sensitivity, PR-curve.
* Use 10-fold cross-validation instead of 4-split evaluation.
* Analyze class imbalance using confusion matrix.

### 3. Explainability

* Integrate Grad-CAM and LIME to visualize regions of interest in CT scans.
* Compare sensitivity maps with heatmaps from Grad-CAM.

### 4. Clinical Validation

* Test model predictions against manual radiologist annotations.
* Conduct feedback sessions with clinical experts.
* Measure usability and diagnostic support.

---

##  Dataset

* Primary: Internal dataset (434 CT images) from original research.
* Secondary: LIDC-IDRI (public dataset of lung CT scans).
* Format: DICOM/PNG → Resized to 512×512 grayscale.

---

##  Key Concepts and Methodologies

| Concept             | Description                             |
| ------------------- | --------------------------------------- |
| CNN                 | For feature extraction from CT images   |
| KNN / FC Classifier | For final decision layer                |
| Transfer Learning   | Reuse pretrained ImageNet weights       |
| Evaluation Metrics  | Accuracy, Sensitivity, AUC, F1-Score    |
| Explainability      | Grad-CAM, LIME for model interpretation |

---

##  Books Used as References

| Book                                                     | Purpose                                         |
| -------------------------------------------------------- | ----------------------------------------------- |
| Deep Learning (Ian Goodfellow)                           | Theory of CNNs, optimization, and training      |
| Deep Learning with Python (François Chollet)             | Practical CNN implementation with Keras         |
| Hands-on ML with Scikit-Learn, Keras, TensorFlow (Géron) | Evaluation metrics, confusion matrix, ROC/AUC   |
| Deep Learning for Medical Image Analysis (Zhou et al.)   | CT image processing, medical CNN models         |
| Explainable AI in Healthcare                             | Grad-CAM, explainability in clinical settings   |
| Data Science for Healthcare                              | Interpretation of metrics in clinical workflows |

---

##  Tools and Libraries

* Python 3.10+
* TensorFlow / PyTorch
* OpenCV, NumPy, Pandas, Scikit-learn
* tf-keras-vis or pytorch-gradcam
* Matplotlib, Seaborn

---

##  Clinical Collaboration Plan

* Step 1: Present model prototype to radiologists (initial validation).
* Step 2: Gather clinical feedback on heatmaps, sensitivity maps.
* Step 3: Adjust model thresholds and evaluate on real-world cases.
* Step 4: Publish findings and release tool for academic/clinical use.

---

##  Expected Outcomes

* Improved and interpretable lung cancer classifier.
* ROC-AUC > 0.95 on both internal and public datasets.
* Clinically usable prototype for radiologist support.

---

##  Contributions

* Literature review and model reproduction
* Model re-design and evaluation enhancement
* Explainability and visualization pipeline
* Clinical integration and feedback loop

---

##  Repository Structure

```
lung-cancer-detection/
├── data/
│   ├── internal_ct/
│   └── lidc_idri/
├── notebooks/
│   ├── 1_preprocessing.ipynb
│   ├── 2_training_baseline_msnn.ipynb
│   ├── 3_gradcam_visualization.ipynb
│   └── 4_evaluation_metrics.ipynb
├── models/
│   └── msnn_v2_weights.h5
├── utils/
│   └── metrics.py
├── requirements.txt
└── README.md
```

---

