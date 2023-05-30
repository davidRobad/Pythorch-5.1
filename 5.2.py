#-----------------------------------
# David Rodriguez Badillo
# Pytorch 5.1
#-----------------------------------
import torch

#------------------------
# Calcular el gradiente 

# Regresión lineal
# f = w*x

# Ejemplo: f = 2*x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)  # Datos de entrada
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)  # Resultados esperados
W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # Coeficiente (peso) inicializado en 0 con requerimiento de cálculo de gradiente

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # Coeficiente (peso) inicializado en 0 con requerimiento de cálculo de gradiente

#---------
# Modelo
#----------
def forward(x):
    return w * x

#--------------------
# Error: Perdida = MSE (Mean Squared Error)
#---------------------
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

print(f'Predicción antes del aprendizaje: f(5) = {forward(5).item():.3f}')

#---------------
# Aprendizaje
#----------------
learning_rate = 0.01  # Tasa de aprendizaje
n_iters = 100  # Número de iteraciones

#---------------------------------------------
for epoch in range(n_iters):
    y_pred = forward(X)  # Realizar la predicción adelante (forward)
    # Calcular el error
    l = loss(Y, y_pred)
    # Calcular el gradiente
    l.backward()
    # Mejorar el coeficiente utilizando el gradiente
    with torch.no_grad():
        w -= learning_rate * w.grad
        # Resetear el gradiente a cero
        w.grad.zero_()

    if epoch % 10 == 0:
        print(f'Época {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')

print(f'Predicción después del aprendizaje: f(5) = {forward(5).item():.3f}')
