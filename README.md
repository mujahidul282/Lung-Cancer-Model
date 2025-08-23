# Lung Cancer Detection using Machine Learning & Deep Learning

This project demonstrates how Machine Learning (ML) and Deep Learning (DL) can be applied for early detection of lung cancer from medical imaging datasets such as CT scans or X-rays.

---

## 🎯 Objective
To build a system that can classify lung scans into categories (e.g., **benign** vs **malignant**) using both ML (traditional feature-based models) and DL (CNN-based models), supporting radiologists in early diagnosis.

---

## 📂 Project Structure
```
.
├── data/                 # Raw and processed data
├── notebooks/            # Jupyter notebooks for EDA & experiments
├── src/                  # Source code
│   ├── ml/               # Machine Learning pipeline
│   ├── dl/               # Deep Learning pipeline
│   ├── utils/            # Helper functions (preprocessing, visualization)
│   └── inference.py      # Script for single-image prediction
├── requirements.txt      # Python dependencies
└── README.md             # Project description
```

---

## 🗃️ Dataset
- Public datasets like **LIDC-IDRI** or **Kaggle Lung Cancer Dataset**.
- Expected structure:
```
data/train/
    class_0/
    class_1/
data/val/
    class_0/
    class_1/
data/test/
    class_0/
    class_1/
```

---

## ⚙️ Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate   # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train Deep Learning Model
```bash
python src/dl/train_dl.py
```

### Train Machine Learning Model
```bash
python src/ml/train_ml.py
```

### Evaluate Model
```bash
python src/dl/eval_dl.py
```

### Inference on Single Image
```bash
python src/inference.py path_to_image.png
```

---

## 📊 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC

---

## 📈 Future Work
- Lung segmentation using U-Net
- 3D CNNs for volumetric CT scans
- Grad-CAM visualizations for explainability
- Hyperparameter optimization

---

## 🤝 Contributing
Contributions are welcome! Please fork the repo and submit a pull request.

---

## 📄 License
This project is licensed under the MIT License.
