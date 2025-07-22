
import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import login
from PIL import Image
import torch

st.set_page_config(page_title="WeCare MedGemma AI", layout="wide")

# Sidebar: WeCare branding & author
with st.sidebar:
    st.image("https://wecarehealth.co.nz/wp-content/uploads/2023/07/wecare_logo.svg", width=180)
    st.markdown("**Created by Dr. John Ko**")
    st.markdown("---")

st.title("🧠 WeCare MedGemma 4B – Medical Image Assistant")
st.markdown("Upload an image (skin, X‑ray, fundus), ask a question, and get an AI-generated analysis powered by Google's MedGemma 4B.")

# Securely login to Hugging Face
try:
    token = st.secrets["HF_TOKEN"]
    login(token=token)
except Exception as e:
    st.error(f"Unable to authenticate with Hugging Face: {e}")
    st.stop()

# File uploader
uploaded = st.file_uploader("📤 Upload medical image", type=["jpg", "jpeg", "png"])
prompt = st.text_input("💬 Ask a question about the image", value="What condition is shown in this image?")

if uploaded and prompt:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Running analysis..."):
        model_id = "google/medgemma-4b-it"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).eval()
        image_payload = processor(prompt, images=image, return_tensors="pt").to(model.device)
        outputs = model.generate(**image_payload, max_new_tokens=512)
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    st.markdown("### 🧠 AI Response")
    st.success(response)
else:
    st.info("Please upload an image and enter a question above.")
