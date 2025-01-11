import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
import os
from datetime import datetime
from app_config import LOCAL_MODEL_PATH, HF_MODEL_PATH, MAX_LENGTH

def get_model_safetensors_date(model_path):
    """Gets the modification date of model.safetensors file in the model directory"""
    if os.path.isfile(model_path):
        # If model_path is directly the file we're looking for
        if os.path.basename(model_path) == "model.safetensors":
            return os.path.getmtime(model_path)
        return None
    
    # Search for model.safetensors in the directory and its subdirectories
    for root, dirs, files in os.walk(model_path):
        if "model.safetensors" in files:
            file_path = os.path.join(root, "model.safetensors")
            return os.path.getmtime(file_path)
    
    return None

@st.cache_resource(ttl=600)  # 600 seconds = 10 minutes
def load_model():
    """Loads the model and tokenizer"""
    try:
        is_cloud = os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud'
        model_version = None
        model_timestamp = None
        
        if not is_cloud and os.path.exists(LOCAL_MODEL_PATH):
            model_path = LOCAL_MODEL_PATH
            local_files = True
            # Get latest file modification timestamp
            timestamp = get_latest_file_date(LOCAL_MODEL_PATH)
            if timestamp:
                model_timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            model_version = f"Local Model"
        else:
            from huggingface_hub import model_info
            hf_info = model_info(HF_MODEL_PATH)
            model_path = HF_MODEL_PATH
            local_files = False
            # Get HuggingFace model version
            model_version = f"HuggingFace - {hf_info.sha[:7]}"
            
            # Try to get the last modified date from the model info
            if hasattr(hf_info, 'last_modified'):
                try:
                    model_timestamp = datetime.fromtimestamp(hf_info.last_modified).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    # If HF timestamp fails, try to get from downloaded files
                    timestamp = get_latest_file_date(model_path)
                    if timestamp:
                        model_timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        model_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        print(f"Loading model from: {model_path}")
        
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
        
        # Store version info in model config
        model.config.model_version = model_version
        model.config.model_timestamp = model_timestamp
        
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e

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
            
            # Version Information
            "Model Version": getattr(model.config, "model_version", "Unknown"),
            "Last Updated": getattr(model.config, "model_timestamp", "Unknown"),
            
            # Tokenizer Information
            "Tokenizer Type": type(tokenizer).__name__,
            "Vocabulary Size (Tokenizer)": len(tokenizer),
            "Model Max Length": tokenizer.model_max_length,
            "Padding Token": tokenizer.pad_token,
            "Unknown Token": tokenizer.unk_token,
            "Special Tokens": {}
        }
        
        # Get special tokens map
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