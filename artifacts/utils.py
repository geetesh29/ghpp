import numpy as np
import pickle
class ghpp():
    def __init__(self,data):
        self.data = data

    def load_model(self):
        with open(r'artifacts/model.pkl','rb') as file:
            self.model = pickle.load(file)


    def predict(self):
        self.load_model()
        MedInc = float(self.data['MedInc'])
        HouseAge = float(self.data['HouseAge'])
        AveRooms = float(self.data['AveRooms'])
        AveBedrms = float(self.data['AveBedrms'])
        Population = float(self.data['Population'])
        AveOccup = float(self.data['AveOccup'])
        Latitude = float(self.data['Latitude'])
        Longitude = float(self.data['Longitude'])

        array = np.array([MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude], ndmin=2)
        print(array)
        print("*"*50)

        res = np.around(self.model.predict(array),2)[0]
        print(res)
        return res
    
if __name__ == "__main__":

    data = {
         'MedInc' : 7.5,
         'HouseAge' : 2.5,
         'AveRooms' : 20.705696,
         'AveBedrms' : 5.071994,
         'Population' : 9856.000000,
         'AveOccup' : 6.582278,
         'Latitude' : 45.750000,
         'Longitude' : -130.800000
    }
    

    
    ghpp_obj = ghpp(data)

    ghpp_obj.predict()

    