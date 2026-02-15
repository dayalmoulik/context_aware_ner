import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from collections import Counter
from src.inference import NERInference
from src.masking import mask_entities, reconstruct_with_highlight

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Context-Aware NER",
    layout="wide",
)

st.title("🔐 Context-Aware Entity Recognition & Sensitivity Masking")

st.markdown(
"""
This application uses a fine-tuned **BERT transformer model**
to detect sensitive entities using contextual understanding.
"""
)

# -----------------------------------
# Sidebar Info
# -----------------------------------
st.sidebar.title("📘 Model Information")
st.sidebar.write("Model: BERT")
st.sidebar.write("Task: Token Classification (BIO)")
st.sidebar.write("Fine-tuned on custom dataset")

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():
    return NERInference("models/bert_ner")

ner = load_model()

# -----------------------------------
# Output Mode Selection
# -----------------------------------
mode = st.radio(
    "Select Output Mode:",
    ["Mask Only", "Highlight Only", "Both"]
)

# -----------------------------------
# Text Input Section
# -----------------------------------
input_text = st.text_area(
    "Enter text:",
    height=150,
    placeholder="Patient Ravi Kumar visited Apollo Hospital in Mumbai."
)

if st.button("Analyze Text"):

    if input_text.strip():

        predictions = ner.predict(input_text)

        col1, col2 = st.columns(2)

        # -----------------------------------
        # Masked Output
        # -----------------------------------
        if mode in ["Mask Only", "Both"]:
            with col1:
                st.subheader("🔒 Masked Output")
                masked_output = mask_entities(predictions)
                st.success(masked_output)

                st.download_button(
                    label="⬇ Download Masked Text",
                    data=masked_output,
                    file_name="masked_output.txt",
                    mime="text/plain"
                )

        # -----------------------------------
        # Highlighted Output
        # -----------------------------------
        if mode in ["Highlight Only", "Both"]:
            with col2:
                st.subheader("🎨 Highlighted Entities")
                highlighted_html, color_map = reconstruct_with_highlight(predictions)
                st.markdown(highlighted_html, unsafe_allow_html=True)

        # -----------------------------------
        # Entity Summary
        # -----------------------------------
        entity_types = [
            label.split("-")[1]
            for _, label in predictions
            if label.startswith("B-")
        ]

        counts = Counter(entity_types)

        if counts:
            st.subheader("📊 Entity Summary")
            for entity, count in counts.items():
                st.write(f"**{entity}** : {count}")

        # -----------------------------------
        # Dynamic Legend
        # -----------------------------------
        if mode in ["Highlight Only", "Both"]:
            st.subheader("🎨 Entity Legend")
            for entity, color in color_map.items():
                st.markdown(
                    f"<span style='background-color:{color};color:white;"
                    f"padding:6px;border-radius:6px;margin-right:8px;'>"
                    f"{entity}</span>",
                    unsafe_allow_html=True
                )

    else:
        st.warning("Please enter some text.")

# -----------------------------------
# File Upload Section
# -----------------------------------
st.subheader("📂 Batch Processing (Upload .txt File)")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    predictions = ner.predict(content)

    st.subheader("🔒 Masked File Output")
    masked_output = mask_entities(predictions)
    st.success(masked_output)

    st.subheader("🎨 Highlighted File Output")
    highlighted_html, color_map = reconstruct_with_highlight(predictions)
    st.markdown(highlighted_html, unsafe_allow_html=True)

    st.subheader("🎨 Entity Legend")
    for entity, color in color_map.items():
        st.markdown(
            f"<span style='background-color:{color};color:white;"
            f"padding:6px;border-radius:6px;margin-right:8px;'>"
            f"{entity}</span>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("Built using a fine-tuned BERT model for context-aware PII detection.")
