#-----------------------------------Importing libraries
from flask import Flask, jsonify, request

from Data_Preprocessing import remove_lines,expand_text,accented_char,clean_data,lemmatization,join_list
import pickle
#-----------------------------------Importing libraries

#-----------------------------------Initialization
app = Flask(__name__)
#-----------------------------------Initialization

#-----------------------------------Config data
count_model = pickle.load(open('Models/count2.pkl', 'rb'))

model = pickle.load(open('Models/model2.pkl', 'rb'))



#-----------------------------------test route
@app.route('/')
def home():
    return jsonify({'response' : 'This is home !'})
#-----------------------------------test route

#-----------------------------------prediction route
@app.route('/predict', methods = ['POST'])
def predict():
    requested_data = request.get_data(as_text = True)

    clean_text_train = remove_lines(requested_data)

    clean_text_train = expand_text(clean_text_train)

    clean_text_train = accented_char(clean_text_train)

    clean_text_train = clean_data(clean_text_train)

    clean_text_train = lemmatization(clean_text_train)

    clean_text_train = join_list(clean_text_train)

    vector = count_model.transform([clean_text_train])
    prediction = model.predict(vector)

    if prediction[0]==0:
        result = "Negative Sentiment"
    elif prediction[0]==1:
        result = "Neutral Sentiment"
    elif prediction[0]==2:
        result = "Positive Sentiment"


    return jsonify({ 'Product Review' : requested_data, 'predictions_made' : result})
#-----------------------------------prediction route

#-----------------------------------run the app
if __name__ == '__main__':
    app.run(port=8080)
#-----------------------------------run the app