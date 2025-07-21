# 🧠 Brain Tumor Detection using MRI & Machine Learning

A machine learning-based project to detect brain tumors from MRI scans using extracted image features. The project includes data preprocessing, feature engineering, multiple ML models, hypothesis testing, and deployment via a Streamlit web app with both image upload and manual input modes.

---

## 🚀 Project Highlights

- ✅ Image feature extraction from brain MRI scans
- ✅ Removed 191 duplicate images & handled imbalance using SMOTE
- ✅ Trained Logistic Regression, SVM, and Random Forest (best)
- ✅ Dimensionality reduction using Truncated SVD
- ✅ Hypothesis testing with p-values for feature significance
- ✅ Dual-mode Streamlit App (image upload + manual input)
- ✅ Real-time prediction with confidence scores

---

## 📁 Folder Structure

```text
📦 Brain-Tumor-ML
├── data/              # MRI dataset (Tumor / No Tumor)
├── notebook/          # Jupyter notebooks for ML & EDA
├── model/             # Saved models (.joblib or .pkl)
├── app.py             # Streamlit web app
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## 📁 Folder Structure


- **Mean Intensity**
- **Texture Variance**
- **Skewness**
- **Edge Sharpness**
- **Symmetry Score**

These were extracted using OpenCV and SciPy from preprocessed grayscale MRI images.

---

## 📊 Model Evaluation

| Model               | Accuracy | F1 Score | Notes                  |
|--------------------|----------|----------|------------------------|
| Logistic Regression| 84%      | 0.83     | Baseline model         |
| SVM                | 88%      | 0.87     | Improved margin        |
| 🔥 Random Forest   | 92%      | 0.91     | ✅ Best performer       |

---

## 📈 Hypothesis Testing Insights

- Tumor images have **higher variance & lower symmetry**
- Mean intensity differs significantly between classes
- Edge sharpness is correlated with tumor presence

---

## 💻 Streamlit App Features

🎯 Choose input method:
- 📷 Upload MRI Image → Auto-extract features
- 🔢 Enter features manually → Predict tumor status

🧪 Output includes:
- Prediction (`Tumor` / `No Tumor`)
- Confidence Score (if available)
- Display of uploaded MRI scan

---

## 🧪 Sample Test Inputs (Manual Mode)

| Feature            | Sample Value |
|--------------------|--------------|
| Mean Intensity     | 104.2        |
| Texture Variance   | 1725.5       |
| Skewness           | 0.45         |
| Edge Sharpness     | 352.3        |
| Symmetry Score     | -18.9        |

---

## ▶️ Run the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py


