from flask import Flask,redirect
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd

lin_data = pd.read_csv('powerproduction.csv')

# drop appropriate rows
lin_data = lin_data.drop([208, 340, 404, 456, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499]);

# X and y values for regression
X = lin_data.iloc[:, 0].values
y = lin_data.iloc[:, 1].values

# The X values are reshaped as 
# they only contain one feature
X = X.reshape(-1, 1)

# Decision Tree regression, input speed
# output predicted power
def decTree(speed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    speed_arr = np.array(speed).reshape(-1, 1)
    return str(regressor.predict(speed_arr)[0])

# Neural Network regression, input speed
# output predicted power
def neuralNet(speed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    input_layer = Input(shape=(X.shape[1],))
    dense_layer_1 = Dense(500, activation='relu')(input_layer)
    dense_layer_2 = Dense(100, activation='relu')(dense_layer_1)
    dense_layer_3 = Dense(50, activation='relu')(dense_layer_2)
    output = Dense(1)(dense_layer_3)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
    model.fit(X_train, y_train, batch_size=2, epochs=50, verbose=0, validation_split=0.2)

    speed_arr = np.array(speed).reshape(-1, 1)
    return str(model.predict(speed_arr)[0][0])


# declare app
app = Flask(__name__)


#@app.route('/')
#def hello():
#    return "Hello World!"

@app.route('/')
def home():
    return redirect("static/index.html")


@app.route("/api/uniform")
def uniform():
    return {"value": decTree(1)}

@app.route("/api/normal")
def normal():
    return {"value": neuralNet(1)}


if __name__ == '__main__':
    app.run()