import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("faces.jpg")
face_cascade = cv2.CascadeClassifier()  
# face_cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml")   
face_cascade.load(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")    
grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
faces = face_cascade.detectMultiScale(grayimg,1.1, 2, 0|cv2.CV_FEATURE_PARAMS_HAAR, (30, 30))   

image_faces = image.copy()
image_faces_blurred = image.copy()
if len(faces) != 0:        
    for (x, y, w, h) in faces:        
        cv2.rectangle(image_faces, (x,y), (x+w,y+h), (0,255,255), 2)                    # Agrego un rectángulo en cada cara
        sub_face = image[y:y+h, x:x+w]
        sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)                              # Aplico borrosidad a cada cara...
        image_faces_blurred[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face    # ... y reemplazo en la imagen

plt.figure()
plt.imshow(cv2.cvtColor(image_faces,cv2.COLOR_BGR2RGB))
plt.figure()
plt.imshow(cv2.cvtColor(image_faces_blurred,cv2.COLOR_BGR2RGB))
plt.show()





