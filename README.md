# Facial Recognition using PCA and Distance-Based Classifiers

## Executive Summary
This project benchmarks classical machine learning approaches for face recognition using Principal Component Analysis (PCA) for dimensionality reduction and distance-based classifiers. The evaluation is performed on the ORL Face Dataset, comparing Nearest Neighbor (1-NN), 3-Nearest Neighbors (3-NN), and Nearest Centroid classifiers.

The objective is to analyze how dimensionality reduction impacts classification performance and to identify an efficient operating point in terms of accuracy and model simplicity.

---

## Project Objectives
- Apply PCA for feature extraction from high-dimensional image data
- Analyze explained variance versus number of principal components
- Compare distance-based classifiers in the reduced feature space
- Evaluate performance using standard multi-class metrics

---

## Dataset
- Name: ORL Face Database
- Size: 40 subjects × 10 images per subject
- Image type: Grayscale facial images
- Preprocessed resolution: 46 × 56 pixels

### Dataset Handling
To avoid repository bloat, the dataset is included as a compressed archive.

### Setup Instructions
```bash
unzip orl.zip
After extraction, the repository structure must be:
Computational-Intelligence/
│── Project.ipynb
│── orl.zip
│── ORL/
│   ├── s1/
│   ├── s2/
│   └── ...
│── README.md

The notebook expects the dataset path to be:
orl_path = "ORL"


