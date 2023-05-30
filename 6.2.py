#------------------------------------
# Rodriguez Badillo David
# Versión 6.2
#------------------------------------

# 1) Modelo de diseño (entrada, salida, pase hacia adelante con diferentes capas)
# 2) Pérdida de construcción y optimizador
# 3) Bucle de entrenamiento
# - Adelante = cálculo de predicción y pérdida
# - Hacia atrás = calcular gradientes
# - Actualizar pesos

import torch
import torch.nn as nn

#-------------------------------------------
# Regresion lineal 
# f = w * x 

# f = 2 * x
#-----------------------------------------------
# 0) Datos para entrenamiento
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)  # Características de entrada
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)  # Resultados esperados

n_samples, n_features = X.shape  # Número de muestras y características
print(f'#muestras: {n_samples}, #características: {n_features}')

# 0) Crear una muestra de prueba
X_test = torch.tensor([5], dtype=torch.float32)

# 1) Diseño del modelo, ¡el modelo debe implementar el pase hacia adelante!
# Aquí podemos usar un modelo incorporado de PyTorch
input_size = n_features
output_size = n_features

# Podemos llamar a este modelo con muestras X
model = nn.Linear(input_size, output_size)

'''
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # Definir diferentes capas
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
'''

'''
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # Definir diferentes capas
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
'''

print(f'Predicción antes del entrenamiento: f(5) = {model(X_test).item():.3f}')

# 2) Definir pérdida y optimizador
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#--------------------------
# 3) Ciclo de entrenamiento
#--------------------------
for epoch in range(n_iters):
    # Predicción = pase hacia adelante con nuestro modelo
    y_predicted = model(X)

    # Pérdida
    l = loss(Y, y_predicted)

    # Calcular gradientes = retropropagación (backward)
    l.backward()
    # Actualizar pesos
    optimizer.step()

    # Resetear gradientes
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()  # Parámetros
        print('época ', epoch+1, ': w = ', w[0][0].item(), ' pérdida = ', l)

print(f'Predicción después del entrenamiento: f(5) = {model(X_test).item():.3f}')
