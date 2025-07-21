import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image
from scipy.stats import skew

# Load trained model
model = joblib.load("best_rf_model.joblib")

# Page setup
st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection App")
st.markdown("Choose how you'd like to provide input for prediction:")

# Option selector
option = st.radio("Select Input Method:", ("📷 Upload MRI Image", "🔢 Enter Numeric Features"))

def extract_features(image):
    image = image.convert("L")
    img = np.array(image.resize((128, 128)))

    mean_intensity = np.mean(img)
    texture_var = np.var(img)
    skewness = skew(img.ravel())
    edge_img = cv2.Laplacian(img, cv2.CV_64F)
    sharpness = edge_img.var()
    flipped = np.fliplr(img)
    symmetry_score = -np.mean(np.abs(img - flipped))

    return [mean_intensity, texture_var, skewness, sharpness, symmetry_score]

def predict_result(features):
    input_array = np.array([features])
    prediction = model.predict(input_array)[0]
    try:
        confidence = round(np.max(model.predict_proba(input_array)[0]) * 100, 2)
    except:
        confidence = "N/A"
    return prediction, confidence

# 📷 Image Upload Option
if option == "📷 Upload MRI Image":
    uploaded_file = st.file_uploader("Upload an MRI scan image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="🧠 Uploaded MRI Image", use_column_width=True)
        img = Image.open(uploaded_file)
        features = extract_features(img)

        if st.button("🚀 Predict from Image"):
            pred, conf = predict_result(features)
            st.markdown("### 🧪 Prediction Result:")
            if pred == 1:
                st.error("⚠️ Tumor Detected")
            else:
                st.success("✅ No Tumor Detected")
            st.markdown(f"**Confidence Score:** `{conf}%`")

# 🔢 Manual Input Option
elif option == "🔢 Enter Numeric Features":
    st.markdown("Enter the following MRI features manually:")
    feature_labels = [
        "Mean Intensity",
        "Texture Variance",
        "Skewness",
        "Edge Sharpness",
        "Symmetry Score"
    ]
    input_values = []
    for label in feature_labels:
        val = st.number_input(f"{label}", value=0.0)
        input_values.append(val)

    if st.button("🚀 Predict from Manual Input"):
        pred, conf = predict_result(input_values)
        st.markdown("### 🧪 Prediction Result:")
        if pred == 1:
            st.error("⚠️ Tumor Detected")
        else:
            st.success("✅ No Tumor Detected")
        st.markdown(f"**Confidence Score:** `{conf}%`")
