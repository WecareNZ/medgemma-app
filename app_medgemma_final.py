
import streamlit as st
from transformers import pipeline
from huggingface_hub import login
from PIL import Image
import torch

# 1. Page config and branding
st.set_page_config(page_title="WeCare MedGemma AI", layout="wide")
with st.sidebar:
    st.image("https://wecarehealth.co.nz/wp-content/uploads/2023/07/wecare_logo.svg", width=180)
    st.markdown("**Created by Dr. John Ko**")
    st.markdown("---")

# 2. Title and instructions
st.title("🧠 WeCare MedGemma 4B – Medical Image Assistant")
st.markdown(
    "Upload a dermatoscope, X‑ray, or fundus image, type your question, "
    "and get an AI-powered analysis via Google’s MedGemma 4B."
)

# 3. Authenticate with Hugging Face
try:
    token = st.secrets["HF_TOKEN"]
    login(token=token)
except Exception as e:
    st.error(f"Authentication failed: {e}")
    st.stop()

# 4. File uploader and prompt input
uploaded = st.file_uploader("📤 Upload medical image", type=["jpg", "jpeg", "png"])
prompt = st.text_input("💬 Ask a question about the image", "What condition is shown in this image?")

if uploaded and prompt:
    # Display the uploaded image
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Lazy-loading and caching the pipeline to avoid reload on each query
    @st.cache_resource(show_spinner=False)
    def load_pipeline():
        return pipeline(
            "image-text-to-text",
            model="google/medgemma-4b-it",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    pipe = load_pipeline()

    # Prepare chat-style messages
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert medical AI assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]},
    ]

    with st.spinner("🔍 Running analysis..."):
        output = pipe(text=messages, max_new_tokens=512)
        # The pipeline returns a list with one item; extract last message content
        response = output[0]["generated_text"][-1]["content"]

    # Display the AI response
    st.markdown("### 🧠 AI Response")
    st.success(response)
else:
    st.info("Please upload a medical image and enter a question to get started.")
