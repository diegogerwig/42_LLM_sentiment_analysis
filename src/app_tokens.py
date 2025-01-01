# app_tokens.py

import streamlit as st

def analyze_tokens(text, tokenizer):
    """Analyzes the tokens in the text"""
    try:
        text = text.strip()
        
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        full_tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        token_analysis = []
        for i, token in enumerate(full_tokens):
            token_id = encoded['input_ids'][0][i].item()
            token_data = {
                'token': token,
                'is_special': token.startswith('['),
                'is_subword': token.startswith('##'),
                'token_id': token_id
            }
            token_analysis.append(token_data)
        
        return token_analysis, len(full_tokens)
    except Exception as e:
        st.error(f"Error in token analysis: {str(e)}")
        return [], 0

def get_token_statistics(token_analysis):
    """Calculate token statistics"""
    if not token_analysis:
        return 0, 0, 0, 0
        
    total = len(token_analysis)
    subwords = sum(1 for t in token_analysis if t['is_subword'])
    special = sum(1 for t in token_analysis if t['is_special'])
    regular = total - subwords - special
    
    return total, regular, subwords, special

def prepare_token_data(token_analysis, attribution_scores=None):
    """Prepare token data for display"""
    token_data = []
    has_attributions = attribution_scores is not None
    
    for i, t in enumerate(token_analysis):
        token_type = "Special" if t['is_special'] else "Subword" if t['is_subword'] else "Word"
        entry = {
            "#": i + 1,
            "Token": t['token'],
            "Type": token_type,
            "ID": t['token_id']
        }
        if has_attributions:
            entry["Influence"] = attribution_scores[i]
        token_data.append(entry)
    
    if has_attributions:
        token_data.sort(key=lambda x: x['Influence'], reverse=True)
    
    return token_data, has_attributions