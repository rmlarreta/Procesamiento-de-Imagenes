import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defininimos función para mostrar imágenes
def imshow(img, title=None, color_img=False, blocking=False):
    plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show(block=False)


files = ["img06.png", "img02.png"]

for f in files:

    #@TODO - BORRAR
    #f = files[0]

    img_ori = cv2.imread(f)
    #imshow(img_ori, 'Img. Original (3 canal)', color_img=True)

    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    #imshow(img_gray, 'Img. Original (1 canal)', color_img=False)

    umbral, thresh_img = cv2.threshold(img_gray, thresh=80, maxval=255, type=cv2.THRESH_BINARY)  # Umbralamos
    #imshow(thresh_img, 'Img. Umbralada')

    # f_blur = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=2)
    # imshow(f_blur, 'Img. Tx')

    # gcan = cv2.Canny(f_blur, threshold1=0.4*255, threshold2=0.75*255)
    # imshow(gcan, "Canny")

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  # Así anda mejor...
    fo = cv2.morphologyEx(thresh_img, kernel=se, op=cv2.MORPH_DILATE)
    imshow(thresh_img, 'Img. Diltada')

    contours, hierarchy = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #print(hierarchy)    # hierarchy: [Next, Previous, First_Child, Parent]

    img_out = cv2.imread(f)
    # _ = cv2.drawContours(img_out, contours, contourIdx=-1, color=(0, 0, 255), thickness=1) 
    # imshow(img_out, 'Contornos', color_img=True)

    for ii, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(contours[ii])
        
        ar = 1.5/3
        if(w/h >= ar*0.6 and w/h<=ar*1.4):
            _ = cv2.rectangle(img_out, (x,y), (x+w,y+h), color=(100, 255, 255), thickness=2)

    imshow(img_out, 'Final', color_img=True)