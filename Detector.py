import cv2
import os
import tensorflow
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


# Cargamos el modelo
model = load_model("modelo.h5")
print(model.summary())
# Etiquetas de las senales 
etiquetas = ['Calzada para automoviles', "Prohibido estacionar", "Prohibido el paso"]


# Capturar el video en tiempo real 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

x1, y1 = 450, 150
x2, y2 = 850, 500

Datos = 'r/Test'
if not os.path.exists(Datos):
    #print('Carpeta creada: ', Datos)
    os.makedirs(Datos)

detect = 0

def detectar():
    ruta_prueba ="r"
    prueba_datagen = ImageDataGenerator(rescale=1./255)
    prueba_generator = prueba_datagen.flow_from_directory(ruta_prueba,target_size=(160,160),color_mode='rgb',class_mode=None)

    lote_test = next(prueba_generator)
    probs = model.predict(lote_test)
    clase = np.argmax(probs, -1)

    if np.max(probs) < 0.85: return 0
    if clase[0] == 0:
        etiqueta = etiquetas[0]
    elif clase[0] == 1:
        etiqueta = etiquetas[1]
    else:
        etiqueta = etiquetas[2]
    # print("Prediccion: ", etiqueta, " clase: ", etiqueta)
    return etiqueta


# While principal 

while True:
    ret, frame = cap.read()
    if ret == False: break
    copia = frame.copy()
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
    # Gurda la imagen del rectangulo 
    senal = copia[y1:y2,x1:x2]
    
    cv2.imshow('frame',frame)
    if detect != 0:
        cv2.putText(frame,detect,(x1,y1-20),2,0.7,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow('frame',frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

    # # Se Activa automaticamente 
    # cv2.imwrite(Datos+'/objeto.jpg', senal)
    # #print('imagen almacenada: ', 'objeto.png')
    # detect = detectar()
    # # print(detect)

    # Activado por una tecla 
    if k == ord('s'):
        cv2.imwrite(Datos+'/objeto.jpg', senal)
        print('imagen almacenada: ', 'objeto.jpg')
        detect = detectar()
        print(detect)

cap.release()
cv2.destroyAllWindows()



   
    
