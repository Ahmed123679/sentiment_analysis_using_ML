import streamlit as st
import sklearn
import pickle
import nltk
from helper import preprocessing_step

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# loading models
model = pickle.load(open('models/model.pkl','rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

# title of the program
st.title('Sentiment Analysis using Machine learning')

# user review
review = st.text_input('Please enter your review')

state = st.button('predict')
#1 preprocessing step
tokens = preprocessing_step(review) 

#2 vectorizer transform step

transformed_data = vectorizer.transform([tokens])

# predict output

output = model.predict(transformed_data)


# predict button

if state:
    st.text(f'output = {'Good Review' if output ==1 else 'Bad Review'}')
