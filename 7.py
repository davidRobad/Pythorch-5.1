#-----------------------------------
# Rodriguez Badillo David
# Versión 7
#-----------------------------------
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#------------------------
# 0) Preparar datos
#------------------------
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

#------------------------------
# 1) Modelo lineal
# Linear model f = wx + b
#------------------------------
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

#---------------------------
# 2) Error y optimizador
#---------------------------
learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

#------------------------------
# 3) Ciclo de aprendizaje
#------------------------------
num_epochs = 100
for epoch in range(num_epochs):
    # Evaluación y cálculo del error
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    #--------------------------------------
    # Cálculo del gradiente y actualización de coeficientes
    #---------------------------------------
    loss.backward()
    optimizer.step()

    # Resetear gradientes
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#------------------------
# Predicción y visualización del resultado
#------------------------
predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')  # Datos reales
plt.plot(X_numpy, predicted, 'b')  # Resultado del modelo
plt.show()
