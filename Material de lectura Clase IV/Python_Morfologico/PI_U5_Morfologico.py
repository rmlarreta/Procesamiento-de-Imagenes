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

# --- Operaciones básicas de conjuntos --------------------------------------------
A = cv2.imread('UTK.tif')
B = cv2.imread('GT.tif')
imshow(A, 'A')
imshow(B, 'B')
Ac = 255 - A
imshow(Ac, 'Ac')

AuB = np.zeros(A.shape)
AuB[np.logical_or(A, B)] = 255
AuB = 255*np.logical_or(A,B)  # Otra opción
imshow(AuB, 'AuB')

AiB = np.zeros(A.shape)
AiB[np.logical_and(A, B)] = 255
AiB = 255*np.logical_and(A,B)  # Otra opción
imshow(AiB, 'AiB')

AmB = np.zeros(A.shape)
AmB[np.logical_and(A, 255-B)] = 255
# AmB = 255*np.logical_and(A, 255-B)   # Otra opción
# AmB = A.copy()  # Otra Opcion
# AmB[B>0]=0      #
imshow(AmB, 'AmB')

plt.close('all')

# --- Operaciones morfológicas ---------------------------------------------------------
# Dilate
F = cv2.imread('broken_text.tif')
imshow(F)
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
Fd = cv2.dilate(F, kernel, iterations=1)
imshow(Fd)

# Erode
F = cv2.imread('wirebond_mask.tif')
imshow(F)
L = 10
# L = 30
# L = 70
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L, L) )
Fe = cv2.erode(F, kernel, iterations=1)
imshow(Fe)

# Open
A = cv2.imread('shapes.tif')
# B = cv2.getStructuringElement(cv2.MORPH_RECT, (37, 37))
B = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
Aop = cv2.morphologyEx(A, cv2.MORPH_OPEN, B)
imshow(A)
imshow(Aop)

# Close
A = cv2.imread('shapes.tif')
B = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
Aclau = cv2.morphologyEx(A, cv2.MORPH_CLOSE, B)
imshow(A)
imshow(Aclau)


# Opening + Closing
f = cv2.imread('fingerprint.tif')
se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
fop = cv2.morphologyEx(f, cv2.MORPH_OPEN, se)
fop_cls = cv2.morphologyEx(fop, cv2.MORPH_CLOSE, se)
imshow(f)
imshow(fop)
imshow(fop_cls)

## Hit or Miss
# ---------------------------------------
# B1 = strel([0 0 0;0 1 1;0 0 0]);
# B2 = strel([1 0 0;1 0 0;1 0 0]);
# g = HIT-or-MISS( f , B1 , B2 );
# ----------------------------------------------
f = cv2.imread('squares.tif', cv2.IMREAD_GRAYSCALE)
B = np.array([[-1, -1, -1], [-1, 1, 1], [-1, 1, 0]])  # "1": lugares donde debe haber hit / "-1":lugares donde debe haber miss /  "0": Da igual
#B = np.array([[-1, 0, 0], [-1, 1, 1], [-1, 0, 0]])
g = cv2.morphologyEx(f, cv2.MORPH_HITMISS, B)
# imshow(f)
# imshow(g)
plt.figure
ax1 = plt.subplot(121)
plt.imshow(f, cmap='gray'), plt.title('Imagen Original')
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(g, cmap='gray'), plt.title('Hit or Miss'), plt.xticks([]), plt.yticks([])
plt.show()

# --- Componentes conectadas -----------------------------------------------------------------------
img = cv2.imread('objects.tif', cv2.IMREAD_GRAYSCALE)
connectivity = 8
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)  # https://docs.opencv.org/4.5.3/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
# --- Otra opcion ----------------
# output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)  # https://docs.opencv.org/4.5.3/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
# # Resultados
# num_labels = output[0]  # Cantidad de elementos
# labels = output[1]      # Matriz con etiquetas
# stats = output[2]       # Matriz de stats
# centroids = output[3]   # Centroides de elementos
# --------------------------------

# Coloreamos los elementos
labels = np.uint8(255/num_labels*labels)
im_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
for centroid in centroids:
    cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
for st in stats:
    cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=2)
imshow(img=im_color, color_img=True)

## --- Reconstrucción Morgológica ---------------------------------------------------------------------
### ImReconstruct
f = cv2.imread('book_text_bw.tif', cv2.IMREAD_GRAYSCALE)
fe = cv2.erode(f, np.ones((51, 1)), iterations=1)
fo = cv2.morphologyEx(f, cv2.MORPH_OPEN, np.ones((51, 1)))
imshow(f)
imshow(fe)
imshow(fo)

def imreconstruct(marker, img):
    B = np.ones((3, 3))
    h = marker
    while True:
        # Dilatar
        dil = cv2.dilate(h, B)
        # Intersecciòn
        hk = np.zeros(dil.shape)
        hk[np.logical_and(dil, img)] = 255
        if np.all(h == hk):
            break
        h = hk
    return h

fobr = imreconstruct(fe, f)
imshow(fobr)

# --- Morfológico en escala de grises ------------------------------------------------------------
plt.close('all')
f = cv2.imread('rice.tif', cv2.IMREAD_GRAYSCALE)
imshow(f)

umbral, g1 = cv2.threshold(f, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)   
imshow(g1)

se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  # Así anda mejor...
fo = cv2.morphologyEx(f, kernel=se, op=cv2.MORPH_OPEN)
imshow(fo)
g2 = cv2.absdiff(f, fo)
imshow(g2)
g3 = cv2.morphologyEx(f, kernel=se, op=cv2.MORPH_TOPHAT)
imshow(g3)
umbral, g4 = cv2.threshold(g2, 50, 255, cv2.THRESH_BINARY)  # https://docs.opencv.org/3.4.15/db/d8e/tutorial_threshold.html   /   https://docs.opencv.org/3.4.15/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
imshow(g4)

plt.figure()
ax1 = plt.subplot(221)
plt.imshow(f, cmap='gray')
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(fo, cmap='gray'), plt.title('Opening'), plt.xticks([]), plt.yticks([])
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(g2, cmap='gray'), plt.title('Imagen Original - Opening'), plt.xticks([]), plt.yticks([])
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(g3, cmap='gray'), plt.title('Top-hat'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure()
ax1 = plt.subplot(221)
plt.imshow(f, cmap='gray')
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(g1, cmap='gray'), plt.title('Umbralado'), plt.xticks([]), plt.yticks([])
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(g3, cmap='gray'), plt.title('Top-hat'), plt.xticks([]), plt.yticks([])
plt.show()
