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
    plt.show(block=blocking)


def thin(f, LIMIT=10000):   # Versión mas "educativa"
    _,fth = cv2.threshold(f,127,255,cv2.THRESH_BINARY)
    skel = np.zeros_like(fth)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    flag=True
    ii = 0
    while flag:
        ii+=1
        temp = cv2.morphologyEx(fth, cv2.MORPH_OPEN, kernel)
        temp = cv2.bitwise_not(temp)
        temp = cv2.bitwise_and(fth, temp)
        skel = cv2.bitwise_or(skel, temp)
        fth = cv2.erode(fth, kernel, iterations=1)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(fth)
        flag = not(maxVal==0)
        if ii==LIMIT:
            print("Warning - Too much iterations... Exit.")
            break
    return skel


def thin_v2(f, LIMIT=10000):    # Versión mas eficiente
    _,fth = cv2.threshold(f,127,255,cv2.THRESH_BINARY)
    skel = np.zeros_like(fth)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    flag=True
    ii = 0
    while flag:
        ii+=1
        eroded = cv2.morphologyEx(fth, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(fth, temp)
        skel = cv2.bitwise_or(skel, temp)
        fth = eroded.copy()
        flag = not(cv2.countNonZero(fth)==0)
        if ii==LIMIT:
            print("Warning - Too much iterations... Exit.")
            break
    return skel

f = cv2.imread('cuadrado.tif', cv2.IMREAD_GRAYSCALE)
# f = cv2.imread('O.png', cv2.IMREAD_GRAYSCALE)
# f = cv2.imread('opencv.png', cv2.IMREAD_GRAYSCALE)
# f = cv2.imread('GT.tif', cv2.IMREAD_GRAYSCALE)
# f = cv2.imread('fingerprint_cleaned.tif', cv2.IMREAD_GRAYSCALE)
# f = cv2.imread('bone.tif', cv2.IMREAD_GRAYSCALE)

# skel = thin(f)
skel = thin_v2(f)

plt.figure()
ax1 = plt.subplot(121)
plt.imshow(f, cmap='gray')
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(skel, cmap='gray'), plt.title('Thin'), plt.xticks([]), plt.yticks([])
plt.show()



