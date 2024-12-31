import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Example reviews
EXAMPLE_REVIEWS = {
    "✨ Positive Example 1": """
    This product exceeded all my expectations! The build quality is exceptional,
    and the performance is outstanding. Customer service was incredibly helpful
    and responsive. Best purchase I've made this year!
    """,
    "🌟 Positive Example 2": """
    Absolutely amazing experience with this product! Setup was a breeze,
    everything works flawlessly, and the value for money is incredible.
    The company clearly cares about quality and customer satisfaction.
    """,
    "💫 Positive Example 3": """
    I'm thoroughly impressed with this purchase. The attention to detail is remarkable,
    the functionality is perfect, and it has made such a positive impact on my daily routine.
    Shipping was fast and the packaging was excellent too.
    """,
    "👎 Negative Example 1": """
    Terrible experience! The product arrived damaged and customer service
    was completely unhelpful. Save your money and avoid this product.
    The quality is much worse than advertised.
    """,
    "❌ Negative Example 2": """
    Don't waste your time with this. It broke within a week of purchase
    and the warranty process is a nightmare. Poor quality control and
    even worse customer support. Very disappointed.
    """,
    "⛔ Negative Example 3": """
    This has been the worst purchase decision ever. The product is unreliable,
    the documentation is misleading, and customer service is non-existent.
    Complete waste of money and time. Would give zero stars if possible.
    """
}

def calculate_background_color(confidence):
    """Calculates background color based on confidence value"""
    # Convert confidence to percentage
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

def display_sentiment(sentiment, confidence):
    """Displays the sentiment analysis results"""
    col1, col2 = st.columns(2)
    
    # Common style for both boxes
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
    
    # Confidence display with dynamic background color
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

def init_session_state():
    """Initialize session state variables"""
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False

@st.cache_resource
def load_model():
    """Loads the model and tokenizer"""
    try:
        model_path = "./models/sentiment_model"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
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
            max_length=512,
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

def calculate_token_attributions(model, tokenizer, text, result):
    """Calculate the attribution of each token to the prediction using integrated gradients"""
    try:
        # Tokenize input
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        # Set model to evaluation mode
        model.eval()
        
        # Get model outputs for attribution
        with torch.no_grad():
            outputs = model(**encoded)
            predicted_class = outputs.logits.argmax(dim=1).item()
        
        # Function to get embedding output
        def get_embedding_output(input_ids):
            return model.distilbert.embeddings.word_embeddings(input_ids)
        
        # Get original embeddings
        embeddings = get_embedding_output(encoded['input_ids'])
        
        # Calculate attributions using input x gradient
        embeddings.retain_grad()
        model.zero_grad()
        
        # Forward pass with gradient calculation
        outputs = model(inputs_embeds=embeddings, attention_mask=encoded['attention_mask'])
        
        # Get score for predicted class
        score = outputs.logits[:, predicted_class]
        
        # Backward pass
        score.backward()
        
        # Get gradients and calculate attribution scores
        gradients = embeddings.grad
        attributions = (gradients * embeddings).sum(dim=-1)
        
        # Normalize attribution scores
        attribution_scores = torch.abs(attributions[0])
        if torch.max(attribution_scores) > 0:
            attribution_scores = attribution_scores / torch.max(attribution_scores)
        
        return attribution_scores.tolist()
    except Exception as e:
        st.error(f"Error calculating attributions: {str(e)}")
        return None

def analyze_tokens(text, tokenizer):
    """Analyzes the tokens in the text"""
    try:
        # Basic text preprocessing
        text = text.strip()
        
        # Get tokens and their IDs
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        # Get full list of tokens including special tokens
        full_tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        # Get vocabulary
        vocab = tokenizer.get_vocab()
        
        # Analyze each token
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

def display_token_analysis(token_analysis, total_tokens, attribution_scores=None):
    """Displays the token analysis results"""
    if not token_analysis:
        return
    
    # Count token types
    subwords = sum(1 for t in token_analysis if t['is_subword'])
    special = sum(1 for t in token_analysis if t['is_special'])
    regular = len(token_analysis) - subwords - special
    total = len(token_analysis)

    # Display token statistics in cards with consistent styling
    st.markdown("### Token Analysis")
    
    # Common style for all stat cards
    card_style = """
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    """
    
    heading_style = """
        margin: 0;
        color: #444;
        font-size: 0.9rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        width: 100%;
    """
    
    value_style = """
        margin: 0.5rem 0;
        font-size: 1.8rem;
        font-weight: bold;
        line-height: 1;
    """
    
    # Create four columns for statistics
    col1, col2, col3, col4 = st.columns(4)
    
    # Total tokens card
    col1.markdown(
        f"""
        <div style='{card_style}'>
        <h4 style='{heading_style}'>Total Tokens</h4>
        <div style='{value_style} color: #0066cc;'>{total_tokens}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Regular words card
    col2.markdown(
        f"""
        <div style='{card_style}'>
        <h4 style='{heading_style}'>Full Words</h4>
        <div style='{value_style} color: #28a745;'>{regular}</div>
        <small style='color: #666;'>({(regular/total*100):.1f}%)</small>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Subwords card
    col3.markdown(
        f"""
        <div style='{card_style}'>
        <h4 style='{heading_style}'>Subwords</h4>
        <div style='{value_style} color: #fd7e14;'>{subwords}</div>
        <small style='color: #666;'>({(subwords/total*100):.1f}%)</small>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Special tokens card
    col4.markdown(
        f"""
        <div style='{card_style}'>
        <h4 style='{heading_style}'>Special Tokens</h4>
        <div style='{value_style} color: #6c757d;'>{special}</div>
        <small style='color: #666;'>({(special/total*100):.1f}%)</small>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display token details with attributions
    st.markdown("### Token Details")
    
    token_data = []
    has_attributions = attribution_scores is not None
    
    # Prepare data
    for i, t in enumerate(token_analysis):
        token_type = "Special" if t['is_special'] else "Subword" if t['is_subword'] else "Word"
        attribution = attribution_scores[i] if has_attributions else 0
        
        entry = {
            "#": i + 1,
            "Token": t['token'],
            "Type": token_type,
            "ID": t['token_id']
        }
        
        if has_attributions:
            entry["Influence"] = attribution
            
        token_data.append(entry)
    
    # Configure columns with better widths
    column_config = {
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
        column_config["Influence"] = st.column_config.NumberColumn(
            "Influence",
            help="Impact on prediction (higher = stronger influence)",
            width=120,
            format="%.3f"
        )
        
        # Sort by influence if available
        token_data.sort(key=lambda x: x['Influence'], reverse=True)
    
    # Display the dataframe with fixed width
    st.dataframe(
        token_data,
        column_config=column_config,
        hide_index=True,
        width=800,
        height=400
    )
    
    if has_attributions:
        # Display influential tokens in two columns
        st.markdown("### Token Influence Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Most Influential Tokens")
            for entry in token_data[:5]:
                influence = entry['Influence']
                token = entry['Token']
                st.markdown(
                    f"""
                    <div style='background-color: rgba(40, 167, 69, 0.1); padding: 0.5rem; 
                    border-radius: 0.3rem; border: 1px solid rgba(40, 167, 69, 0.2); 
                    margin-bottom: 0.5rem;'>
                    <div style='font-size: 1.1rem; color: #28a745;'>{token}</div>
                    <div style='color: #666; font-size: 0.9rem;'>Score: {influence:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        with col2:
            st.markdown("#### Least Influential Tokens")
            for entry in sorted(token_data, key=lambda x: x['Influence'])[:5]:
                influence = entry['Influence']
                token = entry['Token']
                st.markdown(
                    f"""
                    <div style='background-color: rgba(108, 117, 125, 0.1); padding: 0.5rem; 
                    border-radius: 0.3rem; border: 1px solid rgba(108, 117, 125, 0.2); 
                    margin-bottom: 0.5rem;'>
                    <div style='font-size: 1.1rem; color: #6c757d;'>{token}</div>
                    <div style='color: #666; font-size: 0.9rem;'>Score: {influence:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

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

        # Custom CSS for the buttons
        st.markdown("""
        <style>
        div[data-testid="stButton"] > button {
            width: 100%;
            margin-bottom: 8px;
        }
        
        div[data-testid="stButton"] > button[data-testid*="positive"] {
            background-color: rgba(40, 167, 69, 0.2);
            border: 1px solid rgba(40, 167, 69, 0.4);
        }
        
        div[data-testid="stButton"] > button[data-testid*="positive"]:hover {
            background-color: rgba(40, 167, 69, 0.3);
            border: 1px solid rgba(40, 167, 69, 0.5);
        }
        
        div[data-testid="stButton"] > button[data-testid*="negative"] {
            background-color: rgba(220, 53, 69, 0.2);
            border: 1px solid rgba(220, 53, 69, 0.4);
        }
        
        div[data-testid="stButton"] > button[data-testid*="negative"]:hover {
            background-color: rgba(220, 53, 69, 0.3);
            border: 1px solid rgba(220, 53, 69, 0.5);
        }
        </style>
        """, unsafe_allow_html=True)
        
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

def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="wide")
    init_session_state()
    
    try:
        model, tokenizer = load_model()
        
        render_sidebar()
        
        st.title("Sentiment Analysis")
        st.write("Enter your text below to analyze its sentiment.")
        
        # Text input area
        text_input = st.text_area(
            "Enter text to analyze:",
            value=st.session_state.text_input,
            max_chars=15000,
            height=200,
            key="text_area",
            on_change=lambda: setattr(st.session_state, 'analyze_clicked', True)
        )
        
        # Apply custom CSS to increase font size
        st.markdown("""
        <style>
            .stTextArea textarea {
                font-size: 1.2rem;
                line-height: 1.4;
            }
        </style>
        """, unsafe_allow_html=True)
        
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
                    attribution_scores = calculate_token_attributions(model, tokenizer, text_input, result)
                    
                    # Token Analysis
                    token_analysis, total_tokens = analyze_tokens(text_input, tokenizer)
                    if token_analysis:
                        display_token_analysis(token_analysis, total_tokens, attribution_scores)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page. If the error persists, check the model configuration.")

if __name__ == "__main__":
    main()