#-----------------------------------
# David Rodriguez Badillo
# Pytorch 5.1
#-----------------------------------
import numpy as np

#--------------------------------
# Calcular manualmente
#--------------------------------

#--------------------
# Regresion
#--------------------
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

#------------
# Modelo
#-----------
def forward(x):
    return w * x

#-----------------
# Perdida = MSE
#-------------------
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

#---------------------------------
# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
#----------------------------------
def gradient(x, y, y_pred):
    return np.mean(2*x*(y_pred - y))

print(f'Prediccion previa al aprendizaje: f(5) = {forward(5):.3f}')

#--------------------
# Aprendizaje
#--------------------

learning_rate = 0.01  # Coeficiente de aprendizaje
n_iters = 20  # Iteraciones

#----------------------------------------
for epoch in range(n_iters):
    # Prediccion = ADELANTE
    y_pred = forward(X)

    #--------------------
    # Perdida
    #--------------------
    l = loss(Y, y_pred)
    
    # ------------------------
    # Calcula el gradiente
    #---------------------------
    dw = gradient(X, Y, y_pred)

    #  Mejorar coeficiente
    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
     
print(f'Prediccion despues del aprendizaje: f(5) = {forward(5):.3f}')
