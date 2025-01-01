# app_utils.py

import streamlit as st

def calculate_background_color(confidence):
    """Calculates background color based on confidence value"""
    conf_pct = confidence * 100
    
    if conf_pct >= 99.9:  # Effectively 100%
        return "#28a745"  # Green
    elif conf_pct <= 50:
        return "#ffc107"  # Yellow
    else:
        # Calculate color between yellow and green
        ratio = (conf_pct - 50) / 50  # 0 to 1
        # Interpolate between yellow (255, 193, 7) and green (40, 167, 69)
        r = int(255 - (255 - 40) * ratio)
        g = int(193 + (167 - 193) * ratio)
        b = int(7 + (69 - 7) * ratio)
        return f"rgb({r}, {g}, {b})"

def init_session_state():
    """Initialize session state variables"""
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False

def get_column_config(has_attributions=False):
    """Get column configuration for token display"""
    config = {
        "#": st.column_config.NumberColumn(
            "#",
            help="Position in text",
            width=60
        ),
        "Token": st.column_config.TextColumn(
            "Token",
            help="The actual token",
            width=200
        ),
        "Type": st.column_config.TextColumn(
            "Type",
            help="Token type (Word/Subword/Special)",
            width=100
        ),
        "ID": st.column_config.NumberColumn(
            "Token ID",
            help="ID in model's vocabulary",
            width=100
        )
    }
    
    if has_attributions:
        config["Influence"] = st.column_config.NumberColumn(
            "Influence",
            help="Impact on prediction (higher = stronger influence)",
            width=120,
            format="%.3f"
        )
    
    return config