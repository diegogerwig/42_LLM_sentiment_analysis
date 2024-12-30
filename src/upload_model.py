from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get token from environment
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

def upload_model_to_hf(local_model_path, repo_name):
    """
    Uploads the model to Hugging Face Hub
    """
    # First, let's save the model and tokenizer properly
    # Load the base model and tokenizer
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    base_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load your trained weights
    trained_model = AutoModelForSequenceClassification.from_pretrained(
        local_model_path,
        use_safetensors=True
    )
    
    # Save everything properly
    trained_model.save_pretrained(local_model_path)
    base_tokenizer.save_pretrained(local_model_path)
    
    # Initialize Hugging Face API
    api = HfApi()
    api.set_access_token(HF_TOKEN)
    
    # Create or get repository
    repo_url = api.create_repo(
        repo_id=repo_name,
        exist_ok=True,
        private=False
    )
    
    # Upload the files
    api.upload_folder(
        folder_path=local_model_path,
        repo_id=repo_name,
        repo_type="model"
    )
    
    print(f"Model uploaded to: {repo_url}")
    return repo_url

if __name__ == "__main__":
    model_path = "./models/sentiment_model"
    repo_name = "dgerwig/sentiment-analysis"
    
    try:
        repo_url = upload_model_to_hf(model_path, repo_name)
    except Exception as e:
        print(f"Error uploading model: {str(e)}")