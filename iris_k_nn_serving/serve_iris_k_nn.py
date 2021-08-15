from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import datetime
import json
import pickle
import pandas as pd

app = Flask(__name__)
api = Api(app)

df = pd.read_csv('data/IRIS.csv')

x = df.drop(columns=['species', 'target'], axis=1)
y = df.target

# Split the data - 75% train, 25% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Scale the X data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

with open('model/iris_k_nn.pkl', 'rb') as file:
    model = pickle.load(file)

def json_converter(json_data):
    if isinstance(json_data, datetime.datetime):
        return json_data.__str__()

def write_json(data):
    # Serializing json 
    json_object = json.dumps(data)

    output_file = 'output/output.json'
    
    # Writing to sample.json
    with open(output_file, "w") as outfile:
        outfile.write(json_object)

class Testing(Resource):
    def post(self):
        request_data = request.get_json()

        data_testing = {
            'Sepal Length': request_data['Sepal Length'],
            'Sepal Width': request_data['Sepal Width'],
            'Petal Length': request_data['Petal Length'],
            'Petal Width': request_data['Petal Width']
        }

        knn_score = model.score(x_test, y_test)

        knn_predict = model.predict(scaler.transform(pd.DataFrame(data_testing)))
        
        test_result = {
            'timestamp': json_converter(datetime.datetime.now()),
            'method': "NaN",
            'accuracy': 0,
            'predict_result': []
        }

        test_result['method'] = "KNN"
        test_result['accuracy'] = knn_score
        test_result['predict_result'] = json.dumps(knn_predict.tolist())
        write_json(test_result)
        
        return jsonify(test_result)

api.add_resource(Testing, '/testing')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True, port = '5000')

