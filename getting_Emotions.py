from genericpath import exists
from sys import path
from tkinter import Frame
import cv2
import os 
import imutils

#emotionName = 'Felicidad'
#emotionName = 'Enojo'
#emotionName = 'Triste'
emotionName = 'Sorpresa'

folder = 'data'
if not os.path.exists(folder):
    print ('Carpeta creada:',folder)

dataPath = 'C:/Users/gabri/OneDrive/Escritorio/YAIR/Emotion_RecognizerOpenCV/data'
emotionPath = dataPath + '/'+ emotionName

if not os.path.exists(emotionPath):
    print ('Carpeta Creada:', emotionPath)
    os.makedirs(emotionPath)

captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
contador = 0

while True:
    ret ,frame = captura.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=640)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameAux = frame.copy()

    faces = faceClassif.detectMultiScale(gris,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle (frame, (x,y), (x+w,y+h),(0,255,0),2)
        rostro = frameAux[y:y+h, x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(emotionPath + '/rostro_{}.jpg'.format(contador), rostro)
        contador = contador +1
    cv2.imshow('Capturando emociones', frame)

    k =cv2.waitKey(1)
    if k == 27 or contador >=200:
        break
captura.release()
cv2.destroyAllWindows()
