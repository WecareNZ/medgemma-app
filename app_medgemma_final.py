import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
from PIL import Image
import torch

# 1. Page config & branding
st.set_page_config(page_title="WeCare MedGemma AI", layout="wide")
with st.sidebar:
    st.image("https://wecarehealth.co.nz/wp-content/uploads/2023/07/wecare_logo.svg", width=180)
    st.markdown("**Created by Dr. John Ko**")
    st.markdown("---")

# 2. Title & instructions
st.title("🧠 WeCare MedGemma 4B – Medical Image Assistant")
st.markdown(
    "Upload a dermatoscope, X-ray, or fundus image, type your question, "
    "and get an AI-powered analysis via Google’s MedGemma 4B."
)

# 3. Authenticate with Hugging Face
try:
    login(token=st.secrets["HF_TOKEN"])
except Exception as e:
    st.error(f"Authentication failed: {e}")
    st.stop()

# 4. Image uploader & prompt
uploaded = st.file_uploader("📤 Upload medical image", type=["jpg", "jpeg", "png"])
prompt   = st.text_input("💬 Ask a question about the image:", "What condition is shown in this image?")

# 5. Run MedGemma
if uploaded and prompt:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Running analysis..."):
        model_id = "google/medgemma-4b-it"

        # Load processor & model (per HF example) :contentReference[oaicite:0]{index=0}
        processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).eval()

        # Prepare inputs (with the <start_of_image> prefix) :contentReference[oaicite:1]{index=1}
        inputs = processor(
            text=f"<start_of_image> {prompt}",
            images=image,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        # Generate & decode (skip the prompt tokens)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=512)
        gen_tokens = output[0][ inputs["input_ids"].shape[-1] : ]
        response  = processor.decode(gen_tokens, skip_special_tokens=True)

    st.markdown("### 🧠 AI Response")
    st.success(response)

else:
    st.info("Please upload an image and enter a question above.")
