import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from dotenv import load_dotenv
import os
import pandas as pd

# Load environment variables
load_dotenv()

# Get token from environment
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎭",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Loads model from local path"""
    try:
        model_path = "./models/sentiment_model"
        
        # Cargar el modelo y tokenizer localmente
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True
        )
        
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e

def main():
    # Title and description
    st.title("📊 Sentiment Analysis")
    st.write("This application analyzes the sentiment of text using a fine-tuned DistilBERT model.")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.write("""
    This app uses a Deep Learning model to analyze
    text sentiment. The model was trained on review data
    and can classify texts as positive or negative.
    """)
    
    # Main area
    input_option = st.radio(
        "Select input method:",
        ["Single Text", "CSV File"]
    )
    
    if input_option == "Single Text":
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=100,
            placeholder="Write your text here..."
        )
        
        if st.button("Analyze") and text_input:
            with st.spinner("Analyzing..."):
                # Load model and make prediction
                classifier = load_model()
                result = classifier(text_input)[0]
                
                # Show results
                sentiment = "POSITIVE" if result["label"] == "LABEL_1" else "NEGATIVE"
                confidence = result["score"]
                
                col1, col2 = st.columns(2)
                col1.metric("Sentiment", sentiment)
                col2.metric("Confidence", f"{confidence:.2%}")
    
    else:
        # CSV upload
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Select column
            text_column = st.selectbox(
                "Select the column containing text to analyze:",
                df.columns
            )
            
            if st.button("Analyze CSV"):
                with st.spinner("Analyzing texts..."):
                    classifier = load_model()
                    
                    # Process each text
                    results = []
                    for text in df[text_column]:
                        pred = classifier(text)[0]
                        sentiment = "POSITIVE" if pred["label"] == "LABEL_1" else "NEGATIVE"
                        confidence = pred["score"]
                        results.append({"sentiment": sentiment, "confidence": confidence})
                    
                    # Add results to DataFrame
                    results_df = pd.DataFrame(results)
                    df["sentiment"] = results_df["sentiment"]
                    df["confidence"] = results_df["confidence"]
                    
                    # Show results
                    st.write("### Analysis Results")
                    st.dataframe(df)
                    
                    # Download results
                    st.download_button(
                        "Download Results",
                        df.to_csv(index=False).encode('utf-8'),
                        "analysis_results.csv",
                        "text/csv"
                    )
                    
                    # Show statistics
                    st.write("### Statistics")
                    col1, col2 = st.columns(2)
                    pos_pct = (df["sentiment"] == "POSITIVE").mean()
                    col1.metric("Positive Texts", f"{pos_pct:.1%}")
                    col2.metric("Negative Texts", f"{(1-pos_pct):.1%}")

if __name__ == "__main__":
    main()