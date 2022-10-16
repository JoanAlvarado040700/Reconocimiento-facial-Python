.0import cv2 as cv
import numpy as np
import os
import tkinter as tk
from time import time



dataruta = 'D:/Proyectos python/Proyecto reconocimiento facial/data'
listData = os.listdir(dataruta)
entrenamientoEigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
entrenamientoEigenFaceRecognizer.read('EntrenamientoEingenFaceRecognizar.xml')
ruidos = cv.CascadeClassifier('D:\Proyectos python\Entrenamiento de ruidos opencv\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml') 
camara = cv.VideoCapture(0)


while True:
    _,captura = camara.read() 
    gris = cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idcaptura = gris.copy()
    cara = ruidos.detectMultiScale(gris,1.3,5)

    for (x,y,e1,e2) in cara:
        rostroCap = idcaptura[y:y+e2, x:x+e1]
        rostroCap = cv.resize(rostroCap,(160,160),interpolation = cv.INTER_CUBIC)
        resultado = entrenamientoEigenFaceRecognizer.predict(rostroCap)
        cv.putText(captura, '{}'.format(listData[resultado[0]]),(x,y-20), 2,0.7, (0, 255, 0),1,cv.LINE_AA)
        cv.rectangle(captura,(x,y), (x+e1,y+e2), (255, 0, 0), 2)
        '''if resultado[1] < 1000:
            cv.putText(captura, 'No encontrado',(x,y-20), 1,1.1, (0, 255, 0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y),(x+e1,y+e2),(255, 2, 0),2)
        else:
            cv.putText(captura, '{}'.format(listData[resultado[0]]),(x,y-20), 2,0.7, (0, 255, 0),1,cv.LINE_AA)
            cv.rectangle(captura,(x,y), (x+e1,y+e2), (255, 0, 0), 2)
          '''
        
    cv.imshow("Resultados", captura)
    if cv.waitKey(1) == ord('s'):
        break
camara.release()
cv.destroyAllWindows(
