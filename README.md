# medgemma-app

AI image diagnosis app using MedGemma and Streamlit.

## Prerequisites

- Python 3.10 or later
- pip
- (Optional) CUDA-capable GPU for faster inference

## Installation

```bash
git clone <repository-url>
cd medgemma-app
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
```

## Configure Hugging Face Access

The app requires a Hugging Face token to download the MedGemma model.

1. Create a token at [Hugging Face](https://huggingface.co/settings/tokens).
2. Add the token to Streamlit secrets by creating `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "hf_your_token_here"
```

3. (Optional) Export the token as an environment variable for other tools:

```bash
export HF_TOKEN=hf_your_token_here
# or
export HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

## Running the App

Launch the Streamlit server:

```bash
streamlit run app_medgemma_final.py
```

## Features

- Upload dermatoscope, X-ray, or fundus images.
- Ask a question about the image and receive a MedGemma 4B-powered response.
- Uses Hugging Face Hub authentication via `HF_TOKEN`.

