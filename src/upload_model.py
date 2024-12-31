from huggingface_hub import HfApi, login
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get token from environment
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# Login to Hugging Face
login(token=HF_TOKEN)

def upload_model_to_hf(local_model_path, repo_name):
    """
    Uploads the model to Hugging Face Hub
    
    Args:
        local_model_path (str): Path to the local model directory
        repo_name (str): Name of the repository on Hugging Face Hub
        
    Returns:
        str: URL of the uploaded model repository
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