#------------------------------------
# Rodriguez Badillo David
# Versión 7
#-----------------------------------

'''
Las transformaciones se pueden aplicar a imágenes PIL, tensores, ndarrays o datos personalizados
durante la creación del DataSet

lista completa de transformaciones integradas:
https://pytorch.org/docs/stable/torchvision/transforms.html

en imágenes
---------
CenterCrop, Escala de grises, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Redimensionar, Escalar

Sobre tensores
----------
Transformación lineal, normalización, borrado aleatorio

Conversión
----------
ToPILImage: de tensor o ndrarray
ToTensor: de numpy.ndarray o PILImage

Genérico
-------
Usar lambda

Costumbre
------
Escribir clase propia

Componer múltiples transformaciones
---------------------------
compuesta = transforma.Componer([Reescalar(256),
                               Cultivo aleatorio(224)])
'''

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


class WineDataset(Dataset):
    def __init__(self, transform=None):
        # Cargar los datos del archivo CSV del conjunto de datos del vino
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        # Obtener el número de muestras en el conjunto de datos
        self.n_samples = xy.shape[0]
        # Extraer las características (columnas 1 en adelante) y las etiquetas (columna 0) de los datos
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]
        # Almacenar la transformación proporcionada como parámetro
        self.transform = transform

    def __getitem__(self, index):
        # Obtener una muestra específica en la posición 'index'
        sample = self.x_data[index], self.y_data[index]
        # Aplicar la transformación si se ha proporcionado
        if self.transform:
            sample = self.transform(sample)
        # Devolver la muestra
        return sample

    def __len__(self):
        # Devolver el número total de muestras en el conjunto de datos
        return self.n_samples


#--------------------------------------
# Transformaciones comunes
# implementa __call__(self, sample)
#----------------------------------------
class ToTensor:
    def __call__(self, sample):
        # Desempaquetar la muestra en características y etiquetas
        inputs, targets = sample
        # Convertir las características y las etiquetas en tensores de PyTorch
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        # Inicializar el objeto de transformación con un factor específico
        self.factor = factor

    def __call__(self, sample):
        # Desempaquetar la muestra en características y etiquetas
        inputs, targets = sample
        # Multiplicar las características por el factor especificado
        inputs *= self.factor
        # Devolver las características y las etiquetas modificadas
        return inputs, targets


print('Sin transformación')
# Crear una instancia de WineDataset sin ninguna transformación
dataset = WineDataset()
# Obtener la primera muestra del conjunto de datos
first_data = dataset[0]
# Desempaquetar las características y las etiquetas de la primera muestra
features, labels = first_data
# Imprimir el tipo y los valores de las características y las etiquetas
print(type(features), type(labels))
print(features, labels)


print('\nCon Transformación a Tensores')
# Crear una instancia de WineDataset con la transformación ToTensor
dataset = WineDataset(transform=ToTensor())
# Obtener la primera muestra del conjunto de datos
first_data = dataset[0]
# Desempaquetar las características y las etiquetas de la primera muestra
features, labels = first_data
# Imprimir el tipo y los valores de las características y las etiquetas
print(type(features), type(labels))
print(features, labels)

print('\nCon Transformación a Tensores y Multiplicación')
# Combinar la transformación ToTensor y la transformación MulTransform en una composición
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
# Crear una instancia de WineDataset con la composición de transformaciones
dataset = WineDataset(transform=composed)
# Obtener la primera muestra del conjunto de datos
first_data = dataset[0]
# Desempaquetar las características y las etiquetas de la primera muestra
features, labels = first_data