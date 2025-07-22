
import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import login
from PIL import Image
import torch

# — 1. Page config & sidebar branding —
st.set_page_config(page_title="WeCare MedGemma AI", layout="wide")
with st.sidebar:
    st.image("https://wecarehealth.co.nz/wp-content/uploads/2023/07/wecare_logo.svg", width=180)
    st.markdown("**Created by Dr. John Ko**")
    st.markdown("---")

# — 2. Title & instructions —
st.title("🧠 WeCare MedGemma 4B – Medical Image Assistant")
st.markdown(
    "Upload a dermatoscope, normal skin image, type your question, "
    "and get an AI-powered analysis via Google’s MedGemma 4B."
)

# — 3. Authenticate Hugging Face —
try:
    login(token=st.secrets["HF_TOKEN"])
except Exception as e:
    st.error(f"Authentication failed: {e}")
    st.stop()

# — 4. Image uploader & prompt —
uploaded = st.file_uploader("📤 Upload medical image", type=["jpg", "jpeg", "png"])
prompt = st.text_input(
    "💬 Ask a question about the image", 
    value="What condition is shown in this image?"
)

# — 5. Run MedGemma when ready —
if uploaded and prompt:
    # Display the image
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Running analysis..."):
        model_id = "google/medgemma-4b-it"

        # Load processor & model with correct flags
        processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).eval()

        # Prepare inputs and generate
        inputs = processor(
            prompt, 
            images=image, 
            return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Show the AI’s answer
    st.markdown("### 🧠 AI Response")
    st.success(response)

else:
    st.info("Please upload an image and enter a question above.")
