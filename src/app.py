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

def analyze_tokens(text, tokenizer):
    """Analyzes the tokens in the text"""
    try:
        # Basic text preprocessing
        text = text.strip()
        
        # Get original tokens and input IDs
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.encode(text)
        
        # Get vocabulary
        vocab = tokenizer.get_vocab()
        
        # Analyze each token
        token_analysis = []
        for i, token in enumerate(tokens):
            # Get original word from position in text (if possible)
            token_clean = token.replace('##', '')
            
            token_data = {
                'token': token,
                'is_special': token.startswith('['),
                'is_subword': token.startswith('##'),
                'token_id': vocab.get(token, None)
            }
            token_analysis.append(token_data)
        
        return token_analysis, len(input_ids)
    except Exception as e:
        st.error(f"Error in token analysis: {str(e)}")
        return [], 0

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
        - Vocabulary verification
        """)
        
        st.markdown("### 📊 Example Reviews")
        st.write("Click any example to try it:")
        
        for title, text in EXAMPLE_REVIEWS.items():
            if st.button(title, key=f"btn_{title}"):
                st.session_state.text_input = text.strip()
                st.session_state.analyze_clicked = True

def display_sentiment(sentiment, confidence):
    """Displays the sentiment analysis results"""
    col1, col2 = st.columns(2)
    
    # Common style for both boxes
    box_height = "120px"
    
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
    
    # Confidence display with matching height
    confidence_pct = confidence * 100
    confidence_color = "#0066cc"
    col2.markdown(
        f"""
        <div style='padding: 1rem; border-radius: 0.5rem; 
        background-color: #f8f9fa; text-align: center;
        height: {box_height}; display: flex; flex-direction: column; 
        align-items: center; justify-content: center;'>
        <h4 style='margin: 0; color: {confidence_color};'>Confidence</h4>
        <h2 style='margin: 0.5rem 0; color: {confidence_color}; font-size: 2.5rem;'>
            {confidence_pct:.1f}%
        </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_token_analysis(token_analysis, total_tokens):
    """Displays the token analysis results"""
    if not token_analysis:
        return
    
    # Count token types
    subwords = sum(1 for t in token_analysis if t['is_subword'])
    special = sum(1 for t in token_analysis if t['is_special'])
    regular = len(token_analysis) - subwords - special
    
    # Display main token info
    st.markdown(f"### Token Analysis (Total Length: {total_tokens} tokens)")
    
    # Token composition as a simple bar
    st.markdown("#### Token Composition:")
    total = len(token_analysis)
    reg_pct = (regular / total) * 100
    sub_pct = (subwords / total) * 100
    spe_pct = (special / total) * 100
    
    st.markdown(f"""
    - Full Words: {regular} ({reg_pct:.1f}%)
    - Subwords: {subwords} ({sub_pct:.1f}%)
    - Special Tokens: {special} ({spe_pct:.1f}%)
    """)
    
    # Display token details in a compact table
    st.markdown("#### Token Details:")
    col1, col2 = st.columns([2, 3])
    with col1:
        token_data = []
        for i, t in enumerate(token_analysis, 1):
            token_type = "Special" if t['is_special'] else "Subword" if t['is_subword'] else "Word"
            token_data.append({
                "#": i,
                "Token": t['token'],
                "Type": token_type
            })
        
        st.dataframe(
            token_data,
            column_config={
                "#": st.column_config.NumberColumn(
                    "#",
                    width="small",
                ),
                "Token": st.column_config.TextColumn(
                    "Token",
                    width="medium"
                ),
                "Type": st.column_config.TextColumn(
                    "Type",
                    width="small"
                )
            },
            hide_index=True
        )

def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="wide")
    init_session_state()
    
    try:
        model, tokenizer = load_model()
        
        render_sidebar()
        
        st.title("Sentiment Analysis")
        st.write("Enter your text below to analyze its sentiment.")
        
        # Text input area with larger font
        text_input = st.text_area(
            "Enter text to analyze:",
            value=st.session_state.text_input,
            max_chars=15000,
            height=200,
            key="text_area",
            on_change=lambda: setattr(st.session_state, 'analyze_clicked', True),
            help="Write or paste your text here and press Analyze or Ctrl+Enter"
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
                    
                    # Token Analysis
                    token_analysis, total_tokens = analyze_tokens(text_input, tokenizer)
                    if token_analysis:
                        display_token_analysis(token_analysis, total_tokens)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page. If the error persists, check the model configuration.")

if __name__ == "__main__":
    main()