# app_main.py

import streamlit as st
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from app_config import PAGE_TITLE, PAGE_ICON, PAGE_LAYOUT, TEXT_AREA_STYLE
from app_model import load_model, predict_sentiment, calculate_token_attributions
from app_tokens import analyze_tokens
from app_ui import render_sidebar, display_sentiment, display_token_analysis
from app_utils import init_session_state

def main():
    """Main application function"""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=PAGE_LAYOUT)
    init_session_state()
    
    try:
        model, tokenizer = load_model()
        
        render_sidebar()
        
        st.title("Sentiment Analysis")
        st.write("Enter your text below to analyze its sentiment.")
        
        # Add custom CSS for text area
        st.markdown(TEXT_AREA_STYLE, unsafe_allow_html=True)
        
        # Text input area
        text_input = st.text_area(
            "Enter text to analyze:",
            value=st.session_state.text_input,
            max_chars=15000,
            height=200,
            key="text_area",
            on_change=lambda: setattr(st.session_state, 'analyze_clicked', True)
        )
        
        analyze_button = st.button("Analyze")
        
        if (analyze_button or st.session_state.analyze_clicked) and text_input:
            st.session_state.analyze_clicked = False
            
            with st.spinner("Analyzing..."):
                # Sentiment Analysis
                result = predict_sentiment(model, text_input, tokenizer)
                if result:
                    st.markdown("## Results")
                    display_sentiment(result["label"], result["score"])
                    
                    # Calculate token attributions
                    attribution_scores = calculate_token_attributions(model, tokenizer, text_input)
                    
                    # Token Analysis
                    token_analysis, total_tokens = analyze_tokens(text_input, tokenizer)
                    if token_analysis:
                        display_token_analysis(token_analysis, total_tokens, attribution_scores)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page. If the error persists, check the model configuration.")

if __name__ == "__main__":
    main()