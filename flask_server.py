from flask import Flask,redirect,request
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd

lin_data = pd.read_csv('powerproduction.csv')

# drop appropriate rows, they are outliers
lin_data = lin_data.drop([208, 340, 404, 456, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499])

# X and y values for regression
X = lin_data.iloc[:, 0].values
y = lin_data.iloc[:, 1].values

# The X values are reshaped as 
# they only contain one feature
X = X.reshape(-1, 1)

# Decision Tree regression, input speed
# output predicted power
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
decregressor = DecisionTreeRegressor()
decregressor.fit(X_train, y_train)

# return string version
def decTree(speed):
    speed_arr = np.array(speed).reshape(-1, 1)
    return str(decregressor.predict(speed_arr)[0])

# Neural Network regression, input speed
# output predicted power
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=0)

# create neural network: output, input and hidden alyers
input_layer = Input(shape=(X.shape[1],))
dense_layer_1 = Dense(500, activation='relu')(input_layer)
dense_layer_2 = Dense(100, activation='relu')(dense_layer_1)
dense_layer_3 = Dense(50, activation='relu')(dense_layer_2)
output = Dense(1)(dense_layer_3)

# compile and fit model
model = Model(inputs=input_layer, outputs=output)
model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.fit(X_train2, y_train2, batch_size=2, epochs=15, verbose=0, validation_split=0.2)

# return string version
def neuralNet(speed):
    speed_arr = np.array(speed).reshape(-1, 1)
    return str(model.predict(speed_arr)[0][0])


# declare app
app = Flask(__name__)


#@app.route('/')
#def hello():
#    return "Hello World!"

# redirects to main index page at root
@app.route('/')
def home():
    return redirect("static/index.html")


# global speed variable
speed = 0

# this gets speed value from HTML file
# performs dec tree regression
@app.route("/api/decTree", methods = ["GET", "POST"])
def uniform():
    global speed
    if request.method == "POST":
        speed = float(request.json)
    return {"value": decTree(speed)}

# this gets speed value from HTML file
# performs neural net regression
@app.route("/api/neuralNet", methods = ["GET", "POST"])
def normal():
    global speed
    if request.method == "POST":
        speed = float(request.json)
    return {"value": neuralNet(speed)}

# run app
if __name__ == '__main__':
    app.run()