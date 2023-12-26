import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defininimos fuinción para mostrar imágenes
def imshow(img, title=None, color_img=False, blocking=False):
    plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show(block=blocking)


# --- Rotaciones -------------------------------------------------------------------
#  --> Tutorial: https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html

# Cargo Imagen
# img = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('mammogram.tif', cv2.IMREAD_GRAYSCALE)
rows,cols = img.shape
imshow(img, title="Imagen Original")


M = cv2.getRotationMatrix2D((cols/2,rows/2), 90, 1)
# M = cv2.getRotationMatrix2D((cols/2,rows/2), 45, 1)
img_rot_1 = cv2.warpAffine(img, M, (cols,rows))
img_rot_2 = cv2.warpAffine(img, M, (cols*2,rows*2))
imshow(img_rot_1, title="Imagen Rotada")
imshow(img_rot_2, title="Imagen Rotada - Tamaño duplicado")

plt.figure()
ax1 = plt.subplot(221)
plt.xticks([]), plt.yticks([]), plt.imshow(img, cmap='gray'), plt.title('Imagen Original')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img_rot_1,cmap='gray'), plt.title('Imagen Rotada')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img_rot_2,cmap='gray'), plt.title('Imagen Rotada - Tamaño duplicado')
plt.show(block=False)


# --- Transformaciones afines ---------------------------------------------------
# ==============================================================================
# En una transformación afín, todas las líneas paralelas de la imagen original 
# seguirán siendo paralelas en la imagen de salida. 
# Para encontrar la matriz de transformación, necesitamos tres puntos de la 
# imagen de entrada y sus correspondientes ubicaciones en la imagen de salida. 
# Entonces cv2.getAffineTransform creará una matriz de 2x3 que se 
# pasará a cv2.warpAffine.
# ==============================================================================

# Cargo Imagen
img = cv2.imread('lines.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rows,cols,_ = img.shape
imshow(img, title="Imagen Original")

# Transformación Afín
pts1 = np.float32([[50,50],[200,50],[50,200]])      # Defino 3 puntos en la imagen de entrada...
pts2 = np.float32([[10,100],[200,50],[100,250]])    # Defino sus correspondientes posiciones en la imagen de salida...
M = cv2.getAffineTransform(pts1,pts2)               # Obtengo la matriz de la transformación afín.
img_warp = cv2.warpAffine(img,M,(cols,rows))
# img_warp = cv2.warpAffine(img,M,(2*cols,2*rows))

# Agrego los puntos
cv2.circle(img, tuple(pts1[0]), radius=5, color=(255,0,0), thickness=-1)
cv2.circle(img, tuple(pts1[1]), radius=5, color=(255,0,0), thickness=-1)
cv2.circle(img, tuple(pts1[2]), radius=5, color=(255,0,0), thickness=-1)

cv2.circle(img_warp, tuple(pts2[0]), radius=5, color=(255,0,0), thickness=-1)
cv2.circle(img_warp, tuple(pts2[1]), radius=5, color=(255,0,0), thickness=-1)
cv2.circle(img_warp, tuple(pts2[2]), radius=5, color=(255,0,0), thickness=-1)

# Muestro
plt.figure()
plt.subplot(121), plt.imshow(img, cmap="gray"), plt.title('Input')
plt.subplot(122), plt.imshow(img_warp, cmap="gray"), plt.title('Output')
plt.show(block=False)


# --- Otro ejemplo: Espejado ----------------------------------
# Cargo Imagen
img = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
rows,cols = img.shape
pts1 = np.float32([[0,0],[cols-1,0],[0,rows-1]])     
pts2 = np.float32([[cols-1,0],[0,0],[cols-1,rows-1]])
M = cv2.getAffineTransform(pts1,pts2)               
img_warp = cv2.warpAffine(img,M,(rows,cols))
plt.figure()
plt.subplot(121), plt.imshow(img, cmap="gray"), plt.title('Input')
plt.subplot(122), plt.imshow(img_warp, cmap="gray"), plt.title('Output')
plt.show(block=False)


# --- Homography --------------------------------------------------
img = cv2.imread('libro.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rows,cols,_ = img.shape
imshow(img, title="Imagen Original")

# Obtengo puntos de los extremos del libro
pts_src = np.array([[333,82],[771,166],[665,775],[183,669]])  # sup-izq | sup-der | inf-der | inf-izq

# Obtengo puntos destino
ancho = int(np.sqrt(np.sum(np.power(pts_src[0]-pts_src[1],2))))
alto = int(np.sqrt(np.sum(np.power(pts_src[1]-pts_src[2],2))))
pts_dst = np.array([[0,0],[ancho-1,0],[ancho-1,alto-1],[0,alto-1]])  # sup-izq | sup-der | inf-der | inf-izq

# Aplico Homography
h, status = cv2.findHomography(pts_src, pts_dst)
im_dst = cv2.warpPerspective(img, h, (ancho,alto))
im_dst.shape

# Muestro 
plt.figure()
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(im_dst), plt.title('Output')
plt.show(block=False)

