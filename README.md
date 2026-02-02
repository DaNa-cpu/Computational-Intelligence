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


## Methodology

### Preprocessing
- Convert images to grayscale
- Resize to 46 × 56 pixels
- Flatten images into 2576-dimensional vectors
- Standardize features (zero mean, unit variance)

### Dimensionality Reduction
- PCA trained on standardized training data
- Number of components selected based on cumulative explained variance (83%–99.5%)
- Empirical analysis identifies **m = 78** components as an effective operating point

### Classifiers
| Model | Description |
|------|------------|
| Nearest Neighbor (1-NN) | Classification by minimum Euclidean distance |
| 3-Nearest Neighbors (3-NN) | Majority vote over three nearest samples |
| Nearest Centroid | Classification based on class prototypes |

---

## Experimental Setup
- Multiple train/test splits (50/50 to 90/10)
- PCA fitted exclusively on training data
- Classification performed in the PCA-reduced space
- Evaluation on held-out test data

---

## Results

### Optimal Configuration
- PCA components: **m = 78**
- Train/Test split: **50% / 50%**

### Performance Metrics (Macro-Averaged)

| Metric | 1-NN | 3-NN | Nearest Prototype |
|------|------|------|------------------|
| Accuracy | 0.94 | 0.84 | 0.92 |
| Precision | 0.95 | 0.89 | 0.93 |
| Recall | 0.93 | 0.84 | 0.92 |
| F1-score | 0.94 | 0.84 | 0.91 |

Nearest Neighbor achieves the highest accuracy, while Nearest Centroid offers competitive performance with reduced model complexity.

---

## Visual Analysis
The notebook generates:
- Explained variance versus number of PCA components
- Accuracy as a function of PCA dimensionality
- Two-dimensional PCA embeddings
- Eigenfaces visualization
- Confusion matrices for all classifiers

---

## Tech Stack
- Python 3
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- PIL
- Jupyter Notebook

---

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn pillow
