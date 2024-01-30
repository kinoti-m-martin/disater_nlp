from fastapi import FastAPI
import uvicorn
import nltk
import warnings
warnings.filterwarnings("ignore")
#nltk.download("punkt")
#nltk.download("stopwords")
#nltk.download("wordnet")
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import pickle 
import joblib

# Initialze my app
app = FastAPI()

# Home directory
@app.get("/")
def home_dir():
    return print("Welcome to Disaster Text Detection System")

# Load my model
with open("my_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the vectorizer
vectorizer = joblib.load("vectorizer")

# Functions
# 1. Removing urls
def clean_text(text):
    # Remove URLs
    text_no_urls = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove other regular expressions (replace with spaces)
    text_cleaned = re.sub(r'[^a-zA-Z\s]', '', text_no_urls)

    return text_cleaned

# 2. Handling punctuations
def text_lower(text):

    text = text.lower()

    exclude = string.punctuation # Punctuations to remove

    for character in exclude:
        text = text.replace(character, "")
        
    return text

# 3. Tokenizing
def tokenizer(text):
    # Function for tokenizing the text
    return word_tokenize(text)

# 4. Handling stopwords
def stopwords_rem(text):
    stop_words = stopwords.words("english") # Stopwords to remove
    
    text = [word for word in text if word not in stop_words]
    return text

# 5. Lemmatization
def word_lemmatizer(text):

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in text]
    return lemmas

# Joining tokens
#train_set["lemmatized"] = train_set["lemmatized"].str.join(" ")

# Prediction directory
@app.post("/predict/")
def text_predict(text:str):
    # Remove urls if any from the text
    text = clean_text(text)

    # Handle punctuations
    text_cleaned = text_lower(text)

    # Tokenizing
    text_tokens = tokenizer(text_cleaned)

    # Handle stopwords 
    text = stopwords_rem(text_tokens)

    # Lemmatization
    lemmas = word_lemmatizer(text)

    # Join tokens into a string
    text_final = " ".join(lemmas)

    # Text vectorization
    vectorizer_ = vectorizer
    txt_vector = vectorizer_.transform([text_final])

    # Pass the data through the model for prediction
    predictions = model.predict(txt_vector)
    prediction = int(predictions[0])
    
    # Conditions
    if prediction == 0:
        return {prediction:"Not a Disaster"}
    
    else:
        return {prediction:"Disaster"}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)   # Run the app