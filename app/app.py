import streamlit as st
import sys
import os

# Absolute path to project root
PROJECT_ROOT = r"C:\Users\admin\BITS\Sem 3\NLPA\context_aware_ner"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.inference import NERInference
from src.masking import mask_entities

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Context-Aware PII Masking",
    layout="wide"
)

st.title("🔐 Context-Aware Entity Recognition & Masking")
st.write(
    "This application identifies sensitive entities using a fine-tuned "
    "BERT-based NER model and masks them based on linguistic context."
)

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return NERInference("models/bert_ner")

ner = load_model()

# -----------------------------
# Text input section
# -----------------------------
st.subheader("📝 Enter Text")

input_text = st.text_area(
    "Type or paste text below:",
    height=150,
    placeholder="Please create a PowerPoint presentation for Casey Dietrich in Stephenville."
)

if st.button("Mask Text"):
    if input_text.strip():
        predictions = ner.predict(input_text)
        masked_output = mask_entities(predictions)

        st.subheader("🔒 Masked Output")
        st.success(masked_output)
    else:
        st.warning("Please enter some text.")

# -----------------------------
# File upload section
# -----------------------------
st.subheader("📂 Upload Text File")

uploaded_file = st.file_uploader(
    "Upload a .txt file for batch processing",
    type=["txt"]
)

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")

    predictions = ner.predict(content)
    masked_output = mask_entities(predictions)

    st.subheader("🔒 Masked File Output")
    st.success(masked_output)

# -----------------------------
# Entity legend
# -----------------------------
st.subheader("🎨 Entity Legend")

st.markdown(
    """
    - 🟦 **FULLNAME** – Person Names  
    - 🟩 **CITY** – Location Names  
    - 🟥 **Other Sensitive Entities** – Contextually detected  
    """
)
