import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
from PIL import Image
import torch

# — 1. Page configuration & branding —
st.set_page_config(page_title="WeCare MedGemma AI", layout="wide")
with st.sidebar:
    st.image(
        "https://wecarehealth.co.nz/wp-content/uploads/2023/07/wecare_logo.svg",
        width=180
    )
    st.markdown("**Created by Dr. John Ko**")
    st.markdown("---")

# — 2. Title & instructions —
st.title("🧠 WeCare MedGemma 4B – Medical Image Assistant")
st.markdown(
    "Upload a dermatoscope, X-ray, or fundus image, type your question, "
    "and get an AI-powered analysis via Google’s MedGemma 4B."
)

# — 3. Authenticate to Hugging Face —
try:
    login(token=st.secrets["HF_TOKEN"])
except Exception as e:
    st.error(f"Authentication failed: {e}")
    st.stop()

# — 4. Image uploader & prompt input —
uploaded = st.file_uploader("📤 Upload medical image", type=["jpg","jpeg","png"])
prompt   = st.text_input(
    "💬 Ask a question about the image:",
    "What condition is shown in this image?"
)

if not (uploaded and prompt):
    st.info("Please upload an image and enter a question above.")
    st.stop()

# — 5. Display image —
try:
    image = Image.open(uploaded)
    image.verify()
    uploaded.seek(0)
    image = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("Invalid image file. Please upload a valid image.")
    st.stop()

st.image(image, caption="Uploaded Image", use_column_width=True)

# — 6. Lazy-load & cache processor + model —
@st.cache_resource(show_spinner=False)
def load_medgemma():
    model_id = "google/medgemma-4b-it"
    # bfloat16 only on GPU; float32 on CPU
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype
    ).eval()
    return processor, model

processor, model = load_medgemma()

# — 7. Build the “chat” input & run inference :contentReference[oaicite:0]{index=0}
with st.spinner("🔍 Running analysis..."):
    # Apply the chat template (adds <start_of_image>, tokenizes)
    inputs = processor.apply_chat_template(
        [
            {"role":"system", "content":[{"type":"text","text":"You are an expert medical AI."}]},
            {"role":"user",   "content":[{"type":"text","text":prompt}, {"type":"image","image":image}]}
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    # Generate and strip off the input prompt tokens
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=512)
    gen_tokens = generated[0, inputs["input_ids"].shape[-1]:]

    # Decode to plain text
    response = processor.decode(gen_tokens, skip_special_tokens=True)

# — 8. Display the AI’s answer —
st.markdown("### 🧠 AI Response")
st.success(response)
