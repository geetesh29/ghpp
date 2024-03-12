from flask import Flask,request,jsonify,render_template
import numpy as np
import pickle
from artifacts.utils import ghpp

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.form
    print(data)
    print("*"*50)
    
    ghpp_obj = ghpp(data) 
    result = ghpp_obj.predict()

    #    MedInc = float(data['MedInc'])
    #    HouseAge = float(data['HouseAge'])
    #    AveRooms = float(data['AveRooms'])
    #    AveBedrms = float(data['AveBedrms'])
    #    Population = float(data['Population'])
    #    AveOccup = float(data['AveOccup'])
    #    Latitude = float(data['Latitude'])
    #    Longitude = float(data['Longitude'])
    
    #    array = np.array([MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude], ndmin=2)
    #    print(array)

    #    with open('model.pkl', 'rb') as file:
    #        model = pickle.load(file)

    #    result = np.round(model.predict(array),2)     
    #    print(result)

    return render_template('index.html', pred = result)

if __name__ == '__main__':
    app.run(debug=True)