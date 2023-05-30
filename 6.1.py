#------------------------------------
# David Rodriguez Badillo
# Versión 6.1
#-------------------------------------

#----------------------------------------------------------------
# 1) Diseñar el modelo(entrada, salida, NN con muchas capas)
# 2) Definir error y optimizador
# 3) Ciclos de aprendizaje
#       - Forward = evaluar, predecir y calcular el error
#       - Backward = calcular gradiente
#       - Mejorar coeficiente
#-----------------------------------------------------------------
import torch
import torch.nn as nn

#---------------------
# Regresión lineal
# f = w * x 
#----------------------
# Ejemplo: f = 2 * x
#-----------------------------

# Datos para entrenamiento
#----------------------------
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

#------------------------------------------
# 1) Diseño de modelo: coeficiente y NN
#------------------------------------------
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x

print(f'Predicción antes del aprendizaje: f(5) = {forward(5).item():.3f}')

# 2) Definir error y optimizador
learning_rate = 0.01
n_iters = 100

#error loss definido en PyTorch
loss = nn.MSELoss()
#optimizador SGD (Stochastic Gradient Descent)
optimizer = torch.optim.SGD([w], lr=learning_rate)

#--------------------------
# 3) Ciclo de aprendizaje
#--------------------------
for epoch in range(n_iters):
    # Paso hacia adelante: Predicción
    y_predicted = forward(X)

    # Cálculo del error (Loss)
    l = loss(Y, y_predicted)

    # Cálculo del gradiente: Paso hacia atrás
    l.backward()

    # Mejorar coeficiente utilizando el optimizador
    optimizer.step()

    # Reiniciar los gradientes a cero
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print('epoch ', epoch+1, ': w = ',float(w), ' loss = ', l.item)

print(f'Predicción después del aprendizaje: f(5) = {forward(5).item():.3f}')