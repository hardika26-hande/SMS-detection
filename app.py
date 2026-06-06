import streamlit as st
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

# Text Preprocessing Function
def transform_text(text):

    text = text.lower()

    text = word_tokenize(text)

    words = []

    # Remove punctuation and special characters
    for word in text:
        if word.isalnum():
            words.append(word)

    # Remove stopwords
    filtered_words = []

    for word in words:
        if word not in stopwords.words('english'):
            filtered_words.append(word)

    # Stemming
    stemmed_words = []

    for word in filtered_words:
        stemmed_words.append(ps.stem(word))

    return " ".join(stemmed_words)


# Load Saved Model and Vectorizer
@st.cache_resource
def load_files():

    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    return model, vectorizer


model, vectorizer = load_files()


# Streamlit UI
st.title("📩 SMS Spam Detection")

st.write("Enter an SMS message and check whether it is Spam or Ham.")

user_message = st.text_area(
    "Enter SMS Message",
    height=150
)


# Prediction
if st.button("Check Spam"):

    if user_message.strip() == "":
        st.warning("Please enter a message.")

    else:

        processed_message = transform_text(user_message)

        vector_input = vectorizer.transform([processed_message])

        prediction = model.predict(vector_input)

        if prediction[0] == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Ham Message")