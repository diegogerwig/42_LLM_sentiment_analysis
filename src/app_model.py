# app_model.py

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
from app_config import MODEL_PATH, MAX_LENGTH

@st.cache_resource
def load_model():
    """Loads the model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        
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
            prediction = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities)
        
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