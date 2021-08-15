import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('data/IRIS.csv')

x = df.drop(columns=['species', 'target'], axis=1)
y = df.target

# Split the data - 75% train, 25% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Scale the X data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Testing to see results from sklearn.neighbors.KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, p=1)
model.fit(x_train, y_train)

y_pred_test = model.predict(x_test)

print(f"Sklearn KNN Accuracy: {accuracy_score(y_test, y_pred_test)}")

# Save to file in the current working directory
pkl_filename = "model/iris_k_nn.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
    
# Calculate the accuracy score and predict target values
score = pickle_model.score(x_test, y_test)
print(f"Test score: {format(100 * score)}")

# Print the prediction result
predict_Y = pickle_model.predict(x_test)
print(f"The prdiction result: {predict_Y}")
