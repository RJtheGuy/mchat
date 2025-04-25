from huggingface_hub import hf_hub_download
import os

def download_mistral_model():
    repo_id = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    filename = "mistral-7b-instruct-v0.1.Q2_K.gguf"  # Change to desired quant type
    local_dir = "model"


    os.makedirs(local_dir, exist_ok=True)

    print(f"📥 Downloading {filename} from {repo_id} to '{local_dir}'...")
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False  # ensure full copy
    )
    print(f"✅ Model downloaded to: {model_path}")

if __name__ == "__main__":
    download_mistral_model()
