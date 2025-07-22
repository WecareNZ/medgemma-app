
import streamlit as st
from transformers import AutoProcessor, VisionEncoderDecoderModel
from huggingface_hub import login
from PIL import Image
import torch

st.set_page_config(page_title="WeCare MedGemma AI", layout="wide")

# Sidebar: WeCare branding and creator
with st.sidebar:
    st.image("https://wecarehealth.co.nz/wp-content/uploads/2023/07/wecare_logo.svg", width=200)
    st.markdown("**Created by Dr. John Ko**")
    st.markdown("---")

st.title("🧠 WeCare MedGemma 4B - Medical Image Assistant")
st.markdown("Upload a medical image (e.g., skin, X-ray, fundus), ask a question, and get an AI-generated answer using Google's MedGemma 4B.")

# ✅ Secure Hugging Face token stored in Streamlit secrets
try:
    token = st.secrets["HF_TOKEN"]
    login(token=token)
except Exception as e:
    st.error(f"Hugging Face login failed: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("📤 Upload medical image", type=["jpg", "jpeg", "png"])

# Prompt input
prompt = st.text_input("💬 Your question about the image:", "What condition is shown in this image?")

# Run MedGemma
if uploaded_file and prompt:
    image = Image.open(uploaded_file).convert("RGB")

    # Load model
    model_id = "google/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with st.spinner("🔍 Analyzing image..."):
        inputs = processor(prompt, images=image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown("### 🧠 AI Response:")
    st.success(response)
else:
    st.info("Upload an image and enter a question to get started.")
