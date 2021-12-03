import cv2 as cv
import numpy as np
ruidos = cv.CascadeClassifier('D:\Proyectos python\Entrenamiento de ruidos opencv\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml')

camara = cv.VideoCapture(0) #es para abrir la camara 

while True:
    _,captura = camara.read()
    grises = cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    cara = ruidos.detectMultiScale(grises,1.2,5)
    for (x,y,e1,e2) in cara:
        cv.rectangle(captura,(x,y),(x+e1,y+e2),(0, 0, 255),2)
    
    cv.imshow("Resultados", captura)

    if cv.waitKey(1) == ord('s'):
        break
camara.release()
cv.destroyAllWindows()


