import cv2 as cv
import numpy as np
import os  #--> esta libreria sirve para guardar archivos de tipo python




modelo = 'FotosElonMusk'
ruta1 = 'D:/Proyectos python/Proyecto reconocimiento facial'
rutaCompleta = ruta1 + '/' + modelo
if not os.path.exists(rutaCompleta):
    os.makedirs(rutaCompleta)






camara = cv.VideoCapture('ElonMusk.mp4') #es para abrir la camara
ruidos = cv.CascadeClassifier('D:\Proyectos python\Entrenamiento de ruidos opencv\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml') 
id = 0
while True:
    respuesta,captura = camara.read()
    if respuesta == False:break
    grises = cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idcaptura = captura.copy()
    cara = ruidos.detectMultiScale(grises,1.2,5)


    for (x,y,e1,e2) in cara:
        cv.rectangle(captura,(x,y),(x+e1,y+e2),(0, 0, 255),2)
        rostroCap = idcaptura[y:y+e1, x:x+e2]
        rostroCap = cv.resize(rostroCap,(160,160),interpolation = cv.INTER_CUBIC)
        cv.imwrite(rutaCompleta+'/imagen_{}.jpg'.format(id),rostroCap)
        id= id+1



    cv.imshow("Resultados", captura)

    if id == 351 :
        break
camara.release()
cv.destroyAllWindows()


