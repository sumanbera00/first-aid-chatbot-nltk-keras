from flask import Flask, render_template, request, jsonify
from tkinter import messagebox
import pandas as pd
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import random
import json

app = Flask(__name__)

# Load hospital data
hospital_data = pd.read_csv('hospital_directory.csv', dtype={'Pincode': str})

# Load chatbot model and preprocessing artifacts
intents = json.loads(open('Intents.json').read())
chatbot_model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()


# Function to find nearest hospital
def find_nearest_hospital(pincode):
    try:
        # Filter hospitals based on the entered pincode
        hospitals = hospital_data[hospital_data['Pincode'] == pincode]

        if len(hospitals) == 0:
            return "No hospitals found for the entered pincode."
        else:
            nearest_hospitals = hospitals['Hospital_Name'].tolist()
            return "\n".join(nearest_hospitals)
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Function to handle chatbot interaction
def chatbot_response(text):
    ints = predict_class(text, chatbot_model)
    res = getResponse(ints, intents)
    return res


# Preprocessing functions for chatbot
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')


# Route for hospital lookup
@app.route('/lookup', methods=['POST'])
def lookup():
    pincode = request.form['pincode']
    result = find_nearest_hospital(pincode)
    return jsonify({'result': result})


# Route for chatbot interaction
@app.route('/chatbot', methods=['POST'])
def chatbot():
    message = request.form['message']
    response = chatbot_response(message)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
