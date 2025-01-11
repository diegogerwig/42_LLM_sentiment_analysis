# app_ui.py

import streamlit as st
from app_config import EXAMPLE_REVIEWS
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
        - Model information display
        """)
        
        st.markdown("### 📊 Example Reviews")
        st.write("Click any example to try it:")
        
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

def display_token_analysis(token_analysis, total_tokens, attribution_scores=None):
    """Displays the token analysis results"""
    if not token_analysis:
        return
    
    # Calculate statistics
    subwords = sum(1 for t in token_analysis if t['is_subword'])
    special = sum(1 for t in token_analysis if t['is_special'])
    regular = len(token_analysis) - subwords - special
    total = len(token_analysis)
    
    st.markdown("### Token Analysis")
    
    # Display token statistics
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        (col1, "Total Tokens", total_tokens, "#0066cc", total_tokens/total),
        (col2, "Full Words", regular, "#28a745", regular/total),
        (col3, "Subwords", subwords, "#fd7e14", subwords/total),
        (col4, "Special Tokens", special, "#6c757d", special/total)
    ]
    
    for col, title, value, color, percentage in stats:
        col.markdown(
            f"""
            <div style='background-color: #ccc; padding: 0.75rem; border-radius: 0.5rem;
            text-align: center; height: 120px; display: flex; flex-direction: column;
            justify-content: space-between; align-items: center;'>
                <h3 style='margin: 0; padding: 0; color: {color}; font-size: 1.3rem;'>{title}</h3>
                <div style='margin: 0; padding: 0; font-size: 1.6rem; color: {color};'>{value}</div>
                <div style='margin: 0; padding: 0;'>
                    {f"<small style='color: #444;'>({percentage*100:.1f}%)</small>" if percentage is not None else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Display token details
    st.markdown("### Token Details")
    display_token_details(token_analysis, attribution_scores)

def display_token_details(token_analysis, attribution_scores=None):
    """Displays detailed token information"""
    token_data = []
    for i, t in enumerate(token_analysis):
        entry = {
            "#": i + 1,
            "Token": t['token'],
            "Type": "Special" if t['is_special'] else "Subword" if t['is_subword'] else "Word",
            "ID": t['token_id']
        }
        if attribution_scores:
            entry["Influence"] = attribution_scores[i]
        token_data.append(entry)
    
    if attribution_scores:
        token_data.sort(key=lambda x: x['Influence'], reverse=True)
    
    # Configure dataframe columns
    column_config = {
        "#": st.column_config.NumberColumn("#", width=60),
        "Token": st.column_config.TextColumn("Token", width=200),
        "Type": st.column_config.TextColumn("Type", width=100),
        "ID": st.column_config.NumberColumn("Token ID", width=100)
    }
    
    if attribution_scores:
        column_config["Influence"] = st.column_config.NumberColumn(
            "Influence",
            help="Impact on prediction",
            width=120,
            format="%.3f"
        )
    
    # Display dataframe
    st.dataframe(
        token_data,
        column_config=column_config,
        hide_index=True,
        width=800,
        height=400
    )
    
    # Display influence analysis if available
    if attribution_scores:
        st.markdown("### Token Influence Analysis")
        display_influence_analysis(token_data)

def display_influence_analysis(token_data):
    """Displays most and least influential tokens"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Most Influential Tokens")
        for entry in token_data[:5]:
            st.markdown(
                f"""
                <div style='background-color: rgba(40, 167, 69, 0.1); padding: 0.5rem;
                border-radius: 0.3rem; border: 1px solid rgba(40, 167, 69, 0.2);
                margin-bottom: 0.5rem;'>
                <div style='font-size: 1.1rem; color: #28a745;'>{entry['Token']}</div>
                <div style='color: #666; font-size: 0.9rem;'>Score: {entry['Influence']:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with col2:
        st.markdown("#### Least Influential Tokens")
        for entry in sorted(token_data, key=lambda x: x['Influence'])[:5]:
            st.markdown(
                f"""
                <div style='background-color: rgba(108, 117, 125, 0.1); padding: 0.5rem;
                border-radius: 0.3rem; border: 1px solid rgba(108, 117, 125, 0.2);
                margin-bottom: 0.5rem;'>
                <div style='font-size: 1.1rem; color: #6c757d;'>{entry['Token']}</div>
                <div style='color: #666; font-size: 0.9rem;'>Score: {entry['Influence']:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

def display_model_info(model_info):
    """Displays model information with version details"""
    st.markdown("## Model Information")
    
    st.markdown(
        f"""
        <div style='
            padding: 1rem 1.5rem;
            background-color: #2d3748;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            color: white;
        '>
            <div style='
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 1rem;
                margin-bottom: 0.75rem;
            '>
                <div>
                    <h3 style='color: #a0aec0; font-size: 0.9rem; margin-bottom: 0.5rem;'>
                        Model Version
                    </h3>
                    <div style='
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                        background: rgba(255, 255, 255, 0.1);
                        padding: 0.5rem;
                        border-radius: 0.25rem;
                    '>
                        <span style='color: #ffffff; font-family: monospace;'>
                            {getattr(model_info, 'model_version', 'v1.0.0')}
                        </span>
                    </div>
                </div>
                <div>
                    <h3 style='color: #a0aec0; font-size: 0.9rem; margin-bottom: 0.5rem;'>
                        Last Updated
                    </h3>
                    <div style='
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                        background: rgba(255, 255, 255, 0.1);
                        padding: 0.5rem;
                        border-radius: 0.25rem;
                    '>
                        <span style='color: #ffffff;'>
                            {getattr(model_info, 'model_timestamp', datetime.now(timezone(timedelta(hours=1))).strftime('%Y-%m-%d %H:%M:%S'))}
                        </span>
                    </div>
                </div>
            </div>
            <div style='
                border-top: 1px solid rgba(255, 255, 255, 0.1);
                padding-top: 0.75rem;
            '>
                <h3 style='color: #a0aec0; font-size: 0.9rem; margin-bottom: 0.5rem;'>
                    Version Details
                </h3>
                <div style='
                    background: rgba(255, 255, 255, 0.1);
                    padding: 0.75rem;
                    border-radius: 0.25rem;
                '>
                    <div style='color: #ffffff;'>
                        <div style='font-size: 0.9rem; color: #a0aec0; margin-bottom: 0.25rem;'>
                            {getattr(model_info, 'version_notes', 'Sentiment Analysis Model - Initial Release')}
                        </div>
                        <div style='font-size: 0.8rem; margin-top: 0.25rem;'>
                            by <span style='color: #4299e1;'>{getattr(model_info, 'model_author', 'AI Team')}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create tabs for different categories of information
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏗️ Architecture", 
        "📊 Parameters", 
        "🔤 Tokenizer", 
        "⚙️ Configuration"
    ])
    
    with tab1:
        st.markdown("### Model Architecture")
        arch_col1, arch_col2 = st.columns(2)
        
        with arch_col1:
            st.metric("Model Type", model_info["Model Type"])
            st.metric("Hidden Size", model_info["Hidden Size"])
            st.metric("Number of Layers", model_info["Number of Hidden Layers"])
                        
        with arch_col2:
            st.metric("Base Model", model_info["Base Model"])
            st.metric(
                "Attention Heads",
                model_info["Number of Attention Heads"],
                help="Enable model to focus on different parts of the text simultaneously, capturing complex relationships between words.")
    
    with tab2:
        st.markdown("### Model Parameters")
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            st.metric(
                "Total Parameters", 
                f"{model_info['Number of Parameters']:,}"
            )
            
        with param_col2:
            st.metric(
                "Trainable Parameters", 
                f"{model_info['Trainable Parameters']:,}"
            )
            
        # Calculate percentage of trainable parameters
        trainable_percent = (
            model_info['Trainable Parameters'] / 
            model_info['Number of Parameters'] * 100
        )
        
        st.progress(trainable_percent / 100)
        st.caption(f"Trainable parameters: {trainable_percent:.2f}%")
    
    with tab3:
        st.markdown("### Tokenizer Information")
        token_col1, token_col2 = st.columns(2)
        
        with token_col1:
            st.metric("Tokenizer Type", model_info["Tokenizer Type"])
            st.metric("Vocabulary Size", model_info["Vocabulary Size (Tokenizer)"])
            st.metric("Max Length", model_info["Model Max Length"])
            
        with token_col2:
            st.markdown("#### Special Tokens")
            special_tokens = model_info["Special Tokens"]
            for token_name, token_value in special_tokens.items():
                st.code(f"{token_name}: {token_value}")
    
    with tab4:
        st.markdown("### Model Configuration")
        # # Display version information at the top of the configuration tab
        # version_col1, version_col2 = st.columns(2)
        # with version_col1:
        #     st.metric("Model Version", model_info["Model Version"])
        # with version_col2:
        #     st.metric("Last Updated", model_info["Last Updated"])
        
        # st.markdown("---")  # Add separator
        
        # Display other configuration information
        st.json({
            "problem_type": model_info["Problem Type"],
            "num_labels": model_info["Number of Labels"],
            "activation_function": model_info["Activation Function"],
            "max_position_embeddings": model_info["Max Position Embeddings"]
        })