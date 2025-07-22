
import streamlit as st
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login
from PIL import Image
import torch

st.set_page_config(page_title="MedGemma 4B - Medical Image Assistant")

st.title("🧠 MedGemma 4B - Medical Image Assistant")
st.markdown("Upload a medical image (e.g., skin lesion, X-ray, fundus), enter a question, and get an AI-generated answer using Google's MedGemma 4B model.")

# Token input
with st.sidebar:
    st.header("🔐 Hugging Face Token")
    token = st.text_input("Enter your Hugging Face Read token:", type="password")
    if token:
        try:
            login(token=token)
            st.success("Logged in to Hugging Face!", icon="✅")
        except Exception as e:
            st.error(f"Login failed: {e}")

# Upload image
uploaded_file = st.file_uploader("📤 Upload medical image", type=["jpg", "jpeg", "png"])

# Prompt input
prompt = st.text_input("💬 Your question about the image", "What condition is shown in this image?")

# Run MedGemma
if uploaded_file and prompt and token:
    image = Image.open(uploaded_file).convert("RGB")

    # Load model
    model_id = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with st.spinner("🔍 Analyzing image..."):
        inputs = processor(prompt, images=image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown("### 🧠 MedGemma Response:")
    st.success(response)
elif not token:
    st.info("Please enter your Hugging Face token to continue.")
