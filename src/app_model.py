import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
import os
from app_config import LOCAL_MODEL_PATH, HF_MODEL_PATH, MAX_LENGTH
from huggingface_hub import hf_hub_url, model_info, HfApi
from datetime import datetime, timezone, timedelta

def get_last_commit_info(repo_id):
    """
    Gets the last commit information from a Hugging Face repo
    """
    try:
        api = HfApi()
        commits = api.list_repo_commits(repo_id=repo_id)
        
        # Get the latest commit (first in the list)
        if commits:
            last_commit = commits[0]
            return {
                'hash': last_commit.commit_id,
                'date': last_commit.created_at,
                'author': last_commit.author,
                'message': last_commit.title
            }
        return None
        
    except Exception as e:
        print(f"Error accessing commit history: {e}")
        return None

@st.cache_resource(ttl=600)  # 600 seconds = 10 minutes
def load_model():
    """Loads the model and tokenizer"""
    try:
        is_cloud = os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud'

        # Get commit information
        api = HfApi()
        try:
            # Get just the latest commit
            last_commit = api.list_repo_commits(repo_id=HF_MODEL_PATH, max_results=1)[0]
            model_version = f"{last_commit.commit_id[:7]}"
            model_timestamp = last_commit.created_at
        except Exception as e:
            print(f"Error getting commit info: {str(e)}")
            model_version = "Latest version"
            model_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if not is_cloud and os.path.exists(LOCAL_MODEL_PATH):
            model_path = LOCAL_MODEL_PATH
            local_files = True
        else:
            model_path = HF_MODEL_PATH
            local_files = False

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            use_safetensors=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=local_files
        )
        
        # Store version info directly in the model
        model.version = model_version
        model.timestamp = model_timestamp
        
        return model, tokenizer

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e

def get_model_info(model, tokenizer):
    """Gets comprehensive information about the model and tokenizer"""
    try:
        model_info = {
            # Architecture Information
            "Model Type": model.config.model_type,
            "Base Model": model.config._name_or_path,
            "Hidden Size": model.config.hidden_size,
            "Number of Hidden Layers": model.config.num_hidden_layers,
            "Number of Attention Heads": model.config.num_attention_heads,
            "Max Position Embeddings": model.config.max_position_embeddings,
            
            # Model Parameters
            "Number of Parameters": sum(p.numel() for p in model.parameters()),
            "Trainable Parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            
            # Model Configuration
            "Vocabulary Size": model.config.vocab_size,
            "Activation Function": getattr(model.config, "activation", "gelu"),
            "Problem Type": getattr(model.config, "problem_type", "Not specified"),
            "Number of Labels": model.config.num_labels,
            
            # Version Information - Simplified
            "Version": getattr(model, "version", "Unknown"),
            "Last Updated": getattr(model, "timestamp", "Unknown"),
            
            # Tokenizer Information
            "Tokenizer Type": type(tokenizer).__name__,
            "Vocabulary Size (Tokenizer)": len(tokenizer),
            "Model Max Length": tokenizer.model_max_length,
            "Padding Token": tokenizer.pad_token,
            "Unknown Token": tokenizer.unk_token,
            "Special Tokens": {}
        }
        
        if hasattr(tokenizer, 'special_tokens_map'):
            special_tokens = {}
            for key, value in tokenizer.special_tokens_map.items():
                if isinstance(value, str):
                    special_tokens[key] = value
                elif isinstance(value, list):
                    special_tokens[key] = ', '.join(value)
            model_info["Special Tokens"] = special_tokens
        
        return model_info
    except Exception as e:
        st.error(f"Error getting model information: {str(e)}")
        return None

def predict_sentiment(model, text_input, tokenizer):
    """Performs sentiment analysis"""
    try:
        encoded_input = tokenizer(
            text_input,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            output = model(**encoded_input)
            probabilities = F.softmax(output.logits, dim=1)
            confidence = torch.max(probabilities)
            prediction = torch.argmax(probabilities, dim=1)
        
        label = "POSITIVE" if prediction.item() == 1 else "NEGATIVE"
        score = confidence.item()
        
        return {"label": label, "score": score}
    except Exception as e:
        st.error(f"Error in sentiment prediction: {str(e)}")
        return None

def calculate_token_attributions(model, tokenizer, text):
    """Calculate the attribution of each token to the prediction"""
    try:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        model.eval()
        
        with torch.no_grad():
            outputs = model(**encoded)
            predicted_class = outputs.logits.argmax(dim=1).item()
        
        embeddings = model.distilbert.embeddings.word_embeddings(encoded['input_ids'])
        embeddings.retain_grad()
        model.zero_grad()
        
        outputs = model(inputs_embeds=embeddings, attention_mask=encoded['attention_mask'])
        score = outputs.logits[:, predicted_class]
        score.backward()
        
        gradients = embeddings.grad
        attributions = (gradients * embeddings).sum(dim=-1)
        
        attribution_scores = torch.abs(attributions[0])
        if torch.max(attribution_scores) > 0:
            attribution_scores = attribution_scores / torch.max(attribution_scores)
        
        return attribution_scores.tolist()
    except Exception as e:
        st.error(f"Error calculating attributions: {str(e)}")
        return None
