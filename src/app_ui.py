# app_ui.py

import streamlit as st
from app_config import (EXAMPLE_REVIEWS, CARD_STYLE, HEADING_STYLE, 
                       VALUE_STYLE, BUTTON_STYLES)
from app_utils import calculate_background_color

def render_sidebar():
    """Renders the sidebar content"""
    with st.sidebar:
        st.markdown("# 🤖 Sentiment Analyzer")
        
        st.markdown("""
        ### 📝 About
        This tool analyzes text sentiment using a fine-tuned DistilBERT model.
        The model was trained on product reviews and classifies text as positive
        or negative.
        
        ### 🔍 Features
        - Real-time sentiment analysis
        - Token visualization
        - Token influence analysis
        """)
        
        st.markdown("### 📊 Example Reviews")
        st.write("Click any example to try it:")
        
        # Add custom button styles
        st.markdown(BUTTON_STYLES, unsafe_allow_html=True)
        
        # Group examples by sentiment
        st.markdown("##### Positive Examples:")
        for title, text in EXAMPLE_REVIEWS.items():
            if "Positive" in title:
                if st.button(title, key=f"btn_positive_{title}", help="Click to load this positive example"):
                    st.session_state.text_input = text.strip()
                    st.session_state.analyze_clicked = True
        
        st.markdown("##### Negative Examples:")
        for title, text in EXAMPLE_REVIEWS.items():
            if "Negative" in title:
                if st.button(title, key=f"btn_negative_{title}", help="Click to load this negative example"):
                    st.session_state.text_input = text.strip()
                    st.session_state.analyze_clicked = True

def display_sentiment(sentiment, confidence):
    """Displays the sentiment analysis results"""
    col1, col2 = st.columns(2)
    
    box_height = "100px"
    
    # Sentiment display
    sentiment_color = "#28a745" if sentiment == "POSITIVE" else "#dc3545"
    col1.markdown(
        f"""
        <div style='padding: 1rem; border-radius: 0.5rem; 
        background-color: {sentiment_color}; color: white; 
        text-align: center; height: {box_height}; 
        display: flex; align-items: center; justify-content: center;'>
        <h2 style='margin: 0; font-size: 2.5rem;'>{sentiment}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Confidence display
    confidence_pct = confidence * 100
    bg_color = calculate_background_color(confidence)
    text_color = "#000000" if confidence <= 0.5 else "#ffffff"
    
    col2.markdown(
        f"""
        <div style='padding: 1rem; border-radius: 0.5rem; 
        background-color: {bg_color}; text-align: center;
        height: {box_height}; display: flex; 
        align-items: center; justify-content: center;'>
        <div style='text-align: center;'>
            <span style='color: {text_color}; font-size: 2.5rem;'>{confidence_pct:.1f}% </span>
            <span style='color: {text_color}; font-size: 1.5rem;'>confidence</span>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_token_stats(col1, col2, col3, col4, total_tokens, regular, subwords, special, total):
    """Displays token statistics in cards"""
    stats = [
        (col1, "Total Tokens", total_tokens, "#0066cc", None),
        (col2, "Full Words", regular, "#28a745", regular/total),
        (col3, "Subwords", subwords, "#fd7e14", subwords/total),
        (col4, "Special Tokens", special, "#6c757d", special/total)
    ]
    
    for col, title, value, color, percentage in stats:
        content = f"""
        <div style='{CARD_STYLE}'>
        <h4 style='{HEADING_STYLE}'>{title}</h4>
        <div style='{VALUE_STYLE} color: {color};'>{value}</div>
        """
        if percentage is not None:
            content += f"<small style='color: #666;'>({percentage*100:.1f}%)</small>"
        content += "</div>"
        col.markdown(content, unsafe_allow_html=True)

def display_token_analysis(token_analysis, total_tokens, attribution_scores=None):
    """Displays the token analysis results"""
    from app_tokens import get_token_statistics
    from app_utils import get_column_config
    
    if not token_analysis:
        return
    
    total, regular, subwords, special = get_token_statistics(token_analysis)
    
    st.markdown("### Token Analysis")
    
    # Display token statistics
    col1, col2, col3, col4 = st.columns(4)
    display_token_stats(col1, col2, col3, col4, total_tokens, regular, subwords, special, total)
    
    # Display token details
    st.markdown("### Token Details")
    display_token_details(token_analysis, attribution_scores)
    
    if attribution_scores:
        st.markdown("### Token Influence Analysis")
        display_token_influence(token_analysis, attribution_scores)

def display_token_details(token_analysis, attribution_scores):
    """Displays detailed token information"""
    from app_tokens import prepare_token_data
    from app_utils import get_column_config
    
    token_data, has_attributions = prepare_token_data(token_analysis, attribution_scores)
    column_config = get_column_config(has_attributions)
    
    st.dataframe(
        token_data,
        column_config=column_config,
        hide_index=True,
        width=800,
        height=400
    )

def display_token_influence(token_analysis, attribution_scores):
    """Displays token influence analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Most Influential Tokens")
        display_influential_tokens(token_analysis, attribution_scores, True)
    
    with col2:
        st.markdown("#### Least Influential Tokens")
        display_influential_tokens(token_analysis, attribution_scores, False)

def display_influential_tokens(token_analysis, attribution_scores, most_influential=True):
    """Displays most/least influential tokens"""
    token_data = []
    for i, t in enumerate(token_analysis):
        token_data.append({
            'token': t['token'],
            'influence': attribution_scores[i]
        })
    
    sorted_data = sorted(token_data, key=lambda x: x['influence'], reverse=most_influential)[:5]
    bg_color = "rgba(40, 167, 69, 0.1)" if most_influential else "rgba(108, 117, 125, 0.1)"
    border_color = "rgba(40, 167, 69, 0.2)" if most_influential else "rgba(108, 117, 125, 0.2)"
    text_color = "#28a745" if most_influential else "#6c757d"
    
    for entry in sorted_data:
        st.markdown(
            f"""
            <div style='background-color: {bg_color}; padding: 0.5rem; 
            border-radius: 0.3rem; border: 1px solid {border_color}; 
            margin-bottom: 0.5rem;'>
            <div style='font-size: 1.1rem; color: {text_color};'>{entry['token']}</div>
            <div style='color: #666; font-size: 0.9rem;'>Score: {entry['influence']:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )