import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk 
import pickle # MUST import pickle for loading

# --- [CRITICAL STEP 1: LOAD MODEL AND VECTORIZER ONCE] ---
vectorizer_filename = 'fitted_vectorizer.pkl'
model_filename = 'trained_model.sav' 
# Load the fitted vectorizer object
try:
    with open(vectorizer_filename, 'rb') as file:
        FITTED_VECTORIZER = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: Vectorizer file '{vectorizer_filename}' not found. Please check your directory.")
    st.stop()
    
# Load the trained ML model object
try:
    with open(model_filename, 'rb') as file:
        TRAINED_MODEL = pickle.load(file) # <-- The model is now loaded here
except FileNotFoundError:
    st.error(f"Error: Model file '{model_filename}' not found. Did you download it from Colab?")
    st.stop()

# NLTK downloads should be cached to avoid running every time
@st.cache_data
def load_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    return stopwords.words('english')

STOP_WORDS = load_nltk_data()
port_stem = PorterStemmer()


# --- [CRITICAL STEP 2: PREPROCESSING FUNCTION] ---
def stemming(content):
    # 'content' is the input string from the user
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    # Use the pre-loaded STOP_WORDS constant
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in STOP_WORDS]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
# ---------------------------------------------------


# (Your CSS and UI code remains here, unchanged)
# 2. Center the Title
st.markdown(
    "<h1 style='text-align: center; color: #1E90FF;'>Sentiment Analysis App</h1>", 
    unsafe_allow_html=True
)

# 3. Create columns (1:4:1 ratio for centering)
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.markdown("### Enter text for sentiment analysis")
    user_input = st.text_area(
        label="Placeholder, use label_visibility instead",
        placeholder="Type your sentence here...", 
        label_visibility="hidden" 
    )
    clicked = st.button("Analyze Sentiment", type="primary")


# --- [CRITICAL STEP 3: EXECUTE PIPELINE ON CLICK] ---
if clicked:
    if user_input:
        
        # 1. Preprocess the string
        user_input_stemmed = stemming(user_input)
        
        # 2. Convert to Feature Vector using the FITTED_VECTORIZER
        #    The transform() method expects a list/iterable of strings, so wrap it in [ ]
        feature_vector = FITTED_VECTORIZER.transform([user_input_stemmed])
        
        # 3. Predict using the model (Uncomment when you load your model)
        prediction = TRAINED_MODEL.predict(feature_vector)[0]
        
        if prediction == 1:
            result = "Positive 😊"
        else:
            result = "Negative 😞"
        st.success("Predicted Sentiment: " + result)
        
    else:
        st.warning("Please enter some text for analysis before clicking the button.")


