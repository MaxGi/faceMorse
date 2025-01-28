from sklearn import svm
from sklearn import linear_model
from sklearn.manifold import TSNE
import numpy as np
import joblib

#Master model class
class Model:
    def __init__(self, model_name, data_key, model_type="classification"):
        self.training_input = []
        self.training_output = []
        self.training_output_model = []
        
        self.data_key = data_key
        self.model_name = model_name
                
        self.model_type = model_type
        self.model = None
                
        self.isFitted = False
    
    def setModel(self, _model="classification"):
        if _model in self.getAvailableModels():
            self.model_type = _model
        else:
            raise ValueError("Invalid model type")
        
    def loadModel(self):
        try:
            self.model = joblib.load(self.model_name + ".pkl")
            print("Model loaded successfully")
        except FileNotFoundError:
            print("Model file not found")
    
    def saveModel(self):
        print("Saving model")
        if self.model is None:
            print("Model not trained")
            return
        joblib.dump(self.model, self.model_name + ".pkl")
    
    def train(self):
        print("Training model in", self.training_input)
        print("Training model out", self.training_output)
        print("Shape", np.shape(self.training_input), np.shape(self.training_output))
        
        if len(self.training_output) == 0:
            print("No training data on", self.model_name)
            return
        
        
        output_data = np.transpose(self.training_output)
        print("num outputs", np.shape(self.training_output))
                
        if self.model_type == "classification":
            self.model = svm.SVC()
        elif self.model_type == "regression":
            self.model = linear_model.LinearRegression()
        
        self.model.fit(self.training_input, output_data)
                            
    
    def predict(self, data):
        flat_prediction_input = np.array(data[self.data_key]).flatten()
        #Pair output data to output paths
        prediction = self.model.predict([flat_prediction_input])
        return prediction

    def rec(self, input, output):    
        flat_input = np.array(input[self.data_key]).flatten()
        self.training_input.append(flat_input)
        self.training_output.append(output)
        print("Training input", self.training_input)
        print("Training output", self.training_output)
    
    def clear(self):
        self.training_input = []
        self.training_output = []
        self.models = []
        
