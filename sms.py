import streamlit as stl
import pickle
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import string


def transform_text(text):
    lem = WordNetLemmatizer()
    text = text.lower()
    text = nltk.word_tokenize(text)

    a = []
    for i in text:
        if i.isalnum():
            a.append(i)

    text = a[:]
    a.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            a.append(i)

    text = a[:]
    a.clear()

    for i in text:
        a.append(lem.lemmatize(i))

    return " ".join(a)


model = pickle.load(open('models1_sms.pkl', 'rb'))
cv = pickle.load(open('vectorz.pkl', 'rb'))
stl.title('Spam Classifier')
input_message = stl.text_area('Enter a message')

if stl.button('Predict'):
    transform_note = transform_text(input_message)
    vector = cv.transform([transform_note])
    result = model.predict(vector)[0]
    if result == 1:
        stl.header('Spam')
    else:
        stl.header('Not Spam')
