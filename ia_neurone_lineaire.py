import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from google.colab import files


# Parametres
a=0.6
b=2
# DATA
x_data = np.linspace(-10, 10, num=100000)
y_data = a * x_data + b + np.random.normal(size=100000)


# Créer Modèle
model = keras.Sequential()
model.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model.compile(loss='mse', optimizer="adam")
# Display Modèle
model.summary()


# Entrainement
model.fit( x_data, y_data, epochs=5, verbose=1 )


# Output 
y_predicted = model.predict(x_data)
# Resultat
plt.scatter(x_data[::500], y_data[::500])
plt.plot(x_data, y_predicted, 'r', linewidth=4)
plt.grid()
plt.show()


# DATA
x_data = np.linspace(-0, 5, num=200)
y_data = 0.5*x_data*np.cos(x_data) + 0.1*np.random.normal(size=200)
print('Data created successfully')


# Output
y_predicted = model.predict(x_data)
# Resultat
plt.scatter(x_data[::1], y_data[::1])
plt.plot(x_data, y_predicted, 'r', linewidth=4)
plt.grid()
plt.show()