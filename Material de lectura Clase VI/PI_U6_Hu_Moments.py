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

# --- Cargo imagen ---------------------------------------
image = cv2.imread("avion_01.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imshow(image)

# --- Momentos de Hu -------------------------------------
moments = cv2.moments(image)
for k, v in moments.items():
    # print(k, v)
    # print(f"{k:4}: {v}")
    # print(f"{k:4}: {v:5.2e}")
    print(f"{k:4}: {v:+5.2e}")

hu_moments = cv2.HuMoments(moments)
hu_moments = hu_moments.flatten()
print(hu_moments)
for ii,vv in enumerate(hu_moments):
    # print(f"{ii:1d}: {vv}")
    print(f"{ii:1d}: {vv:5.2e}")

# ---------------------------------------------------------------------------
# --- Ejemplo con varias imagenes -------------------------------------------
# ---------------------------------------------------------------------------
image = cv2.imread("aviones.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imshow(image)

contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  
len(contours)

image_cnts = cv2.merge((image, image, image))
cv2.drawContours(image_cnts, contours, contourIdx=-1, color=(0, 0, 255), thickness=2)  # https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
imshow(image_cnts)

# --- Visualizo individualmente -----------------------------
c = contours[0]
(x, y, w, h) = cv2.boundingRect(c)
roi = image[y:y + h, x:x + w]
imshow(roi)
hu_moments = cv2.HuMoments(cv2.moments(roi)).flatten()

# --- Obtengo valores para todos los objetos ---------------
result = np.zeros((7,len(contours)))
for ii,c in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = image[y:y + h, x:x + w]
    # imshow(roi)
    hu_moments = cv2.HuMoments(cv2.moments(roi)).flatten()
    result[:,ii] = hu_moments

print(result)