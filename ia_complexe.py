import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import tensorflow as tf
from tensorflow import keras
import math


# DATA
x_data = np.linspace(-0, 5, num=200)
y_data = 0.5*x_data*np.cos(x_data) + 0.1*np.random.normal(size=200)
print('Data created successfully')


# Créer Modèle
model = keras.Sequential()
model.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model.add(keras.layers.Dense(units = 64, activation = 'elu'))
model.add(keras.layers.Dense(units = 64, activation = 'elu'))
model.add(keras.layers.Dense(units = 1, activation = 'linear'))
model.compile(loss='mse', optimizer="adam")
# Display Modèle
model.summary()


# Entrainement
model.fit( x_data, y_data, epochs=300, verbose=1)


# Output
y_predicted = model.predict(x_data)
# Resultat
plt.scatter(x_data[::1], y_data[::1])
plt.plot(x_data, y_predicted, 'r', linewidth=4)
plt.grid()
plt.show()