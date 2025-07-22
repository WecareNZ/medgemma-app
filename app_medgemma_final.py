import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
from PIL import Image
import torch

# 1️⃣ Page setup & branding
st.set_page_config(page_title="WeCare MedGemma AI", layout="wide")
with st.sidebar:
    st.image("https://wecarehealth.co.nz/wp-content/uploads/2023/07/wecare_logo.svg", width=180)
    st.markdown("**Created by Dr. John Ko**")
    st.markdown("---")

st.title("🧠 WeCare MedGemma 4B – Medical Image Assistant")
st.markdown(
    "Upload a dermatoscope, X-ray, or fundus image, type your question, "
    "and get an AI-powered answer via Google’s MedGemma 4B."
)

# 2️⃣ Authenticate to Hugging Face
try:
    login(token=st.secrets["HF_TOKEN"])
except Exception as e:
    st.error(f"Authentication failed: {e}")
    st.stop()

# 3️⃣ Upload image & enter prompt
uploaded = st.file_uploader("📤 Upload medical image", type=["jpg","jpeg","png"])
prompt   = st.text_input("💬 Ask a question about the image:", 
                         "What condition is shown in this image?")

if not (uploaded and prompt):
    st.info("Please upload an image and enter a question above.")
    st.stop()

# 4️⃣ Display the image
image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Uploaded Image", use_column_width=True)

# 5️⃣ Load & cache the processor + model
@st.cache_resource(show_spinner=False)
def load_medgemma():
    model_id = "google/medgemma-4b-it"
    # pick dtype: bfloat16 if GPU, else float32
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype
    ).eval()
    return processor, model

processor, model = load_medgemma()

# 6️⃣ Prepare the “chat” inputs exactly as in the MedGemma docs :contentReference[oaicite:0]{index=0}
messages = [
    {"role":"system", "content":[{"type":"text","text":"You are an expert medical AI."}]},
    {"role":"user",   "content":[{"type":"text","text":prompt}, {"type":"image","image":image}]}
]

with st.spinner("🔍 Running MedGemma..."):
    # apply the chat template, add generation prompt, tokenize, return tensors
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    # run inference
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=512)
    # strip off the input prompt tokens
    gen_tokens = generated[0, inputs["input_ids"].shape[-1]:]
    # decode to text
    response = processor.decode(gen_tokens, skip_special_tokens=True)

# 7️⃣ Show the answer
st.markdown("### 🧠 AI Response")
st.success(response)
