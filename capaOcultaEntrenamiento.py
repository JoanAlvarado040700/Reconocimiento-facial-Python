import cv2 as cv
import numpy as np
import os
import tkinter as tk
from time import time



dataruta = 'D:/Proyectos python/Proyecto reconocimiento facial/data'
listData = os.listdir(dataruta)

IDs = []
rostrosData = []
id = 0
tiempoIni = time()

#--- >proceso para agregar etiquetas a las imagenes y recorido de imagenes
for fila in listData:
    rutaCompleta = dataruta+'/'+fila
    print("Iniciando lectura...")
    for archivo in os.listdir(rutaCompleta):
        print("imagenes: ",fila +'/'+archivo)
        IDs.append(id)
        rostrosData.append(cv.imread(rutaCompleta+'/'+archivo,0))
        #imagenes = cv.imread(rutaCompleta+'/'+archivo,0)
    id = id+1
    tiempoFinal = time()
    tiempoLectura= tiempoFinal - tiempoIni
    print("Tiempo total: ",tiempoLectura)

# ---> modelo de entrenamiento
entrenamientoEigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
print("Iniciando el entrenamiento... espere")
entrenamientoEigenFaceRecognizer.train(rostrosData,np.array(IDs))
tiempoFinalEntrenamiento = time()
totalEntrenamiento = tiempoFinalEntrenamiento - tiempoLectura
print("Tiempo total de entrenamiento: ",totalEntrenamiento)
entrenamientoEigenFaceRecognizer.write("EntrenamientoEingenFaceRecognizar.xml")

print("Entrenamiendo concluido ")
