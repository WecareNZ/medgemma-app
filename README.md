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

## Configure User Authentication

The app uses [streamlit-authenticator](https://github.com/mkhorasani/Streamlit-Authenticator) to require a login before image upload.

1. Generate a hashed password:

   ```bash
   python -c "import streamlit_authenticator as stauth; print(stauth.Hasher(['your_password']).generate()[0])"
   ```

2. Add credentials and cookie settings to `.streamlit/secrets.toml`:

   ```toml
   [credentials.usernames]
   user = {name = "Display Name", password = "<hashed_password>"}

   [cookie]
   name = "medgemma_app"
   key = "some_signature_key"
   expiry_days = 1
   ```

3. Run the app and log in with the username and password.

## Running the App

Launch the Streamlit server:

```bash
streamlit run app_medgemma_final.py
```

## Features

- Upload dermatoscope, X-ray, or fundus images.
- Ask a question about the image and receive a MedGemma 4B-powered response.
- Uses Hugging Face Hub authentication via `HF_TOKEN`.
- Login required before uploading images.

