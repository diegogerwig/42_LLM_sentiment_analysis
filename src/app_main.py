import streamlit as st
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from app_config import PAGE_TITLE, PAGE_ICON, PAGE_LAYOUT, TEXT_AREA_STYLE, BUTTON_STYLES
from app_model import load_model, predict_sentiment, calculate_token_attributions, get_model_info
from app_tokens import analyze_tokens
from app_ui import render_sidebar, display_sentiment, display_token_analysis, display_model_info
from app_utils import init_session_state

def reset_all_states():
    """Reset all relevant session states"""
    keys_to_reset = [
        'text_input',
        'analyze_clicked',
        'analysis_result',
        'token_analysis',
        'attribution_scores'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

def main():
    """Main application function"""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=PAGE_LAYOUT)
    init_session_state()
    
    try:
        # Load model and add button styles
        model, tokenizer = load_model()
        st.markdown(BUTTON_STYLES, unsafe_allow_html=True)
        
        # Render sidebar
        render_sidebar()
        
        st.title("🔮 Sentiment Analysis 🔮")
        st.write("Enter your text below to analyze its sentiment.")
        
        # Add custom CSS for text area
        st.markdown(TEXT_AREA_STYLE, unsafe_allow_html=True)
        
        # Text input area
        text_input = st.text_area(
            "Enter text to analyze:",
            value=st.session_state.get('text_input', ''),
            max_chars=15000,
            height=200,
            key="text_area"
        )
        
        # Update text_input in session state
        if 'text_input' not in st.session_state or st.session_state.text_input != text_input:
            st.session_state.text_input = text_input
            st.session_state.analyze_clicked = False
        
        # Create columns for buttons
        col1, col2 = st.columns(2)
        
        # Analyze button (blue)
        analyze_button = col1.button("ANALYZE", use_container_width=True)
        
        # Clear button (yellow)
        if col2.button("CLEAR", use_container_width=True):
            reset_all_states()
            st.rerun()
        
        if (analyze_button or st.session_state.get('analyze_clicked', False)) and text_input:
            st.session_state.analyze_clicked = True
            
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
        
        # Add separator before model information
        st.markdown("---")
        
        # Always display model information at the end
        model_info = get_model_info(model, tokenizer)
        if model_info:
            display_model_info(model_info)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page. If the error persists, check the model configuration.")

if __name__ == "__main__":
    main()