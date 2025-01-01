# app_config.py

# Example reviews for demonstration
EXAMPLE_REVIEWS = {
    "👍 Positive Example 1": """
    This product exceeded all my expectations! The build quality is exceptional, and the performance is outstanding. Customer service was incredibly helpful and responsive. Best purchase I've made this year!
    """,
    "👍 Positive Example 2": """
    Absolutely amazing experience with this product! Setup was a breeze, everything works flawlessly, and the value for money is incredible. The company clearly cares about quality and customer satisfaction.
    """,
    "👍 Positive Example 3": """
    I'm thoroughly impressed with this purchase. The attention to detail is remarkable, the functionality is perfect, and it has made such a positive impact on my daily routine. Shipping was fast and the packaging was excellent too.
    """,
    "👎 Negative Example 1": """
    Terrible experience! The product arrived damaged and customer service was completely unhelpful. Save your money and avoid this product. The quality is much worse than advertised.
    """,
    "👎 Negative Example 2": """
    Don't waste your time with this. It broke within a week of purchase and the warranty process is a nightmare. Poor quality control and even worse customer support. Very disappointed.
    """,
    "👎 Negative Example 3": """
    This has been the worst purchase decision ever. The product is unreliable, the documentation is misleading, and customer service is non-existent. Complete waste of money and time. Would give zero stars if possible.
    """
}

# Model configuration
MODEL_PATH = "./models/sentiment_model"
MAX_LENGTH = 512

# UI Configuration
PAGE_TITLE = "Sentiment Analysis"
PAGE_ICON = "🎭"
PAGE_LAYOUT = "wide"

# # Styles
# CARD_STYLE = """
#     background-color: #aaa;
#     padding: 1rem;
#     border-radius: 0.5rem;
#     text-align: center;
#     height: 120px;
#     display: flex;
#     flex-direction: column;
#     justify-content: center;
#     align-items: center;
#     overflow: hidden;
# """

# HEADING_STYLE = """
#     margin: 0;
#     color: #444;
#     font-size: 0.9rem;
#     white-space: nowrap;
#     overflow: hidden;
#     text-overflow: ellipsis;
#     width: 100%;
# """

# VALUE_STYLE = """
#     margin: 0.5rem 0;
#     font-size: 1.8rem;
#     font-weight: bold;
#     line-height: 1;
# """

# Text area styling
TEXT_AREA_STYLE = """
<style>
    .stTextArea textarea {
        font-size: 1.2rem;
        line-height: 1.4;
        padding: 0.5rem;
    }
</style>
"""

# Button styling
BUTTON_STYLES = """
<style>
    /* Main buttons styles */
    section[data-testid="stHorizontalBlock"] button {
        height: 6rem !important;
        font-size: 3rem !important;
        font-weight: bold !important;
        padding: 1rem !important;
    }

    section[data-testid="stHorizontalBlock"] > div:first-child button {
        background-color: #2196F3 !important;
        color: white !important;
        border: none !important;
    }

    section[data-testid="stHorizontalBlock"] > div:last-child button {
        background-color: #FFC107 !important;
        color: black !important;
        border: none !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebarContent"] .stButton > button {
        height: auto !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stSidebarContent"] button[data-testid*="btn_positive"] {
        background-color: rgb(40, 167, 69) !important;
        color: white !important;
    }

    [data-testid="stSidebarContent"] button[data-testid*="btn_negative"] {
        background-color: rgb(220, 53, 69) !important;
        color: white !important;
    }
</style>
"""