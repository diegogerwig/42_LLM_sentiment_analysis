from huggingface_hub import HfApi
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get token from environment
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

def upload_model_to_hf(local_model_path, repo_name):
    """
    Uploads the model to Hugging Face Hub
    local_model_path: path to saved model 
    repo_name: repository name in HF 
    """
    api = HfApi()
    
    # Login using token
    api.set_access_token(HF_TOKEN)
    
    # Create repository
    repo_url = api.create_repo(
        repo_id=repo_name,
        exist_ok=True
    )
    
    # Upload model and associated files
    api.upload_folder(
        folder_path=local_model_path,
        repo_id=repo_name,
        repo_type="model"
    )
    
    return repo_url

if __name__ == "__main__":
    # Usage example
    model_path = "./models/sentiment_model"
    repo_name = "dgerwig/sentiment-analysis"
    
    try:
        repo_url = upload_model_to_hf(model_path, repo_name)
        print(f"Model successfully uploaded to: {repo_url}")
    except Exception as e:
        print(f"Error uploading model: {str(e)}")