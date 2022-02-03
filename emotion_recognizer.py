from tkinter import image_names
from unittest import result
import cv2
import os
from cv2 import resize 
import numpy as np
import imutils as im 

def emotionImage(emotion):
    #emoji
    if emotion == 'Felicidad': image =cv2.imread('Emoji/felicidad.jpg')
    if emotion == 'Enojo': image = cv2.imread('Emoji/enojo.jpg')
    if emotion == 'Sorpresa': image = cv2.imread('Emoji/sorpresa.jpg')
    if emotion == 'Tristeza': image = cv2.imread('Emoji/Triteza')
    return image

method = 'EigenFaces'
#method = 'FisherFaces'
#method = 'LBPH'

if method == 'EigenFaces':
    emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if method == 'FisherFaces':
    emotion_recognizer = cv2.face.FisherFacesRecognizer_create()
if method == 'LBPH':
    emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

emotion_recognizer.read('modelo'+method+'.xml')

dataPath = 'C:/Users/gabri/OneDrive/Escritorio/YAIR/Emotion_RecognizerOpenCV/data'
imagePath = os.listdir(dataPath)
print ('imagenPaths: ',imagePath)

captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    ret, frame = captura.read()
    if ret == False:
        break
    gris = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frameAux = gris.copy()

    nframe = cv2.hconcat([frame, np.zeros((480,300,3),dtype=np.uint8)])
    faces = faceClassif.detectMultiScale(gris,1.3,5)

    for (x,y,w,h) in faces:
        rostro = frameAux[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        resultado = emotion_recognizer.predict(rostro)

        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
        
        if method == 'EigenFaces':
            if resultado[1] < 5700:
                cv2.putText(frame,'{}'.format(imagePath[resultado[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                image = emotionImage(imagePath[resultado[0]])
                image2 = im.resize(image, width =480, height=300)
                nframe = cv2.hconcat([frame, image2])
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                nframe = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
        '''
        if method == 'FisherFace':
            if resultado[1] <500:
                cv2.putText(frame,'{}'.format(imagePath[resultado[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+y,y+h),(0,255,0),2)
                image = emotionImage(imagePath[resultado[0]])
                nframe = cv2.hconcat([frame, image])
            else: 
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                nframe = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
        
        if method == 'LBPH':
            if resultado[1] <60:
                cv2.putText(frame,'{}'.format(imagePath[resultado[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                image = emotionImage(imagePath[resultado[0]])
                nframe = cv2.hconcat([frame, image])
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+y,y+h),(0,0,255),2)
                nframe = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
        '''
    cv2.imshow('nframe', nframe)
    k  =cv2.waitKey(1)
    if k == 27:
        break
captura.release()
cv2.destroyAllWindows()





