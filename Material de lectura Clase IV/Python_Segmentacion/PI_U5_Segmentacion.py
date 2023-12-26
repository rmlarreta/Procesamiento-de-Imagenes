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


# --- Cargo Imagen --------------------------------------------------------------
f = cv2.imread('building.tif')              # Leemos imagen
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)  # Pasamos a escala de grises
imshow(gray)

# --- SOBEL ---------------------------------------------------------------------
ddepth = cv2.CV_16S  # Formato salida
grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3) # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3) # Tutorial: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html

# Pasamos a 8 bit
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
cv2.imshow("Sobel dx", abs_grad_x)
cv2.imshow("Sobel dy", abs_grad_y)

# Sumamos los gradientes en una nueva imagen
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
cv2.imshow("Sobel", grad)

# --- Umbralizamos los gradientes -------------------------------
abs_grad_x_th = np.zeros(abs_grad_x.shape) 
abs_grad_x_th[abs_grad_x == abs_grad_x.max()] = 255
cv2.imshow("sobel x + umbral", abs_grad_x_th)

abs_grad_y_th = np.zeros(abs_grad_y.shape) 
abs_grad_y_th[abs_grad_y == abs_grad_y.max()] = 255
cv2.imshow("sobel y + umbral", abs_grad_y_th)

grad_th = np.zeros(grad.shape) 
grad_th[grad >= 0.5*grad.max()] = 255
cv2.imshow("sobel x+y + umbral", grad_th)

# --- LoG ----------------------------------------------------------
blur = cv2.GaussianBlur(gray, (3,3), 0)
LoG = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)
LoG_abs = cv2.convertScaleAbs(LoG)   # Pasamos a 8 bit
imshow(LoG_abs, title="LoG + abs()")
# imshow(np.abs(LoG), title="LoG")
LoG_abs_th = LoG_abs > LoG_abs.max()*0.3
imshow(LoG_abs_th, title="LoG + abs() + umbral")

def Zero_crossing(image):
    z_c_image = np.zeros(image.shape)
    # For each pixel, count the number of positive and negative pixels in the neighborhood
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i+1, j-1],image[i+1, j],image[i+1, j+1],image[i, j-1],image[i, j+1],image[i-1, j-1],image[i-1, j],image[i-1, j+1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1
            # If both negative and positive values exist in the pixel neighborhood, then that pixel is a potential zero crossing
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            # Change the pixel value with the maximum neighborhood difference with the pixel
            if z_c:
                if image[i,j]>0:
                    z_c_image[i, j] = image[i,j] + np.abs(e)
                elif image[i,j]<0:
                    z_c_image[i, j] = np.abs(image[i,j]) + d
                
    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image/z_c_image.max()*255
    z_c_image = np.uint8(z_c_norm)

    return z_c_image

LoG_z = Zero_crossing(LoG)
th = LoG_z.max()*0.3
LoG_zth = np.uint8(LoG_z > th)*255
imshow(LoG_z, title="LoG + Zero Corssing")
imshow(LoG_zth, title="LoG + Zero Corssing + Umbral")


# --- CANNY ---------------------------------------------------------------------------------------
f_blur = cv2.GaussianBlur(f, ksize=(3, 3), sigmaX=1.5)
gcan = cv2.Canny(f_blur, threshold1=0.04*255, threshold2=0.1*255)
gcan = cv2.Canny(f_blur, threshold1=0.4*255, threshold2=0.75*255)
cv2.imshow("Canny", gcan)


# --- Contornos -----------------------------------------------------------------------------------
f = cv2.imread('contornos.png')             # Leemos imagen
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)  # Pasamos a escala de grises
cv2.imshow('contornos', gray)

umbral, thresh_img = cv2.threshold(gray, thresh=128, maxval=255, type=cv2.THRESH_BINARY)  # Umbralamos
cv2.imshow('Umbral', thresh_img)

# Tutorial: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga95f5b48d01abc7c2e0732db24689837b

# Dibujamos
f = cv2.imread('contornos.png')
cv2.drawContours(f, contours, contourIdx=-1, color=(0, 0, 255), thickness=2)  # https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
cv2.imshow('Contornos', f)

# Contornos externos
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# Dibujamos
f = cv2.imread('contornos.png')
cv2.drawContours(f, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)
cv2.imshow('contours externos', f)

# Contornos por jerarquía
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(hierarchy)    # hierarchy: [Next, Previous, First_Child, Parent]

# --- Contornos que no tienen padres ----------------------------------------------
f = cv2.imread('contornos.png')
for ii in range(len(contours)):
    if hierarchy[0][ii][3]==-1:
        cv2.drawContours(f, contours, contourIdx=ii, color=(0, 255, 0), thickness=2)
cv2.imshow('Contornos sin padre', f)

# --- Contornos que no tienen hijos ----------------------------------------------
f = cv2.imread('contornos.png')
for ii in range(len(contours)):
    if hierarchy[0][ii][2]==-1:
        cv2.drawContours(f, contours, contourIdx=ii, color=(0, 255, 0), thickness=2)
cv2.imshow('Contorno sin hijos', f)

# --- Ejemplo particular ---------------------------------------------------------
k = 4 
f = cv2.imread('contornos.png')
cv2.drawContours(f, contours, contourIdx=k, color=(0, 255, 0), thickness=2)
cv2.imshow('Contorno particular', f)
print(hierarchy[0][k])

# Dibujo al padre en azul
if hierarchy[0][k][3] != -1:
    cv2.drawContours(f, contours, contourIdx=hierarchy[0][k][3], color=(255, 0, 0), thickness=2)
cv2.imshow('Contorno particular', f)

# Dibujo los hijos en rojo
for ii in range(len(contours)):
    if hierarchy[0][ii][3]==k:
        cv2.drawContours(f, contours, contourIdx=ii, color=(0, 0, 255), thickness=2)
cv2.imshow('Contorno particular', f)

# Dibujo todos los que están en su mismo nivel
for ii in range(len(contours)):
    if hierarchy[0][ii][3]==hierarchy[0][k][3]:
        cv2.drawContours(f, contours, contourIdx=ii, color=(0, 255, 255), thickness=2)
cv2.imshow('Contorno particular', f)


# --- Ordeno según los contornos mas grandes -------------------------------------
contours_area = sorted(contours, key=cv2.contourArea, reverse=True)
f = cv2.imread('contornos.png')
cv2.drawContours(f, contours_area, contourIdx=0, color=(255, 0, 0), thickness=2)
cv2.drawContours(f, contours_area, contourIdx=1, color=(0, 255, 0), thickness=2)
cv2.drawContours(f, contours_area, contourIdx=2, color=(0, 0, 255), thickness=2)
cv2.imshow('Contorno ordenados por area', f)


# -- Aproximación de contornos con polinomios ----------------------------------
# cnt = contours[2] # Rectángulo
cnt = contours[12]  # Círculo
f = cv2.imread('contornos.png')
cv2.drawContours(f, cnt, contourIdx=-1, color=(255, 0, 0), thickness=2)
cv2.imshow('Aproximacion de contorno', f)

approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)   # https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
len(cnt)
len(approx) 

cv2.drawContours(f, approx, contourIdx=-1, color=(0, 0, 255), thickness=2)
imshow(f)

# Bounding Box
x,y,w,h = cv2.boundingRect(cnt)
f = cv2.imread('contornos.png')
cv2.drawContours(f, cnt, contourIdx=-1, color=(255, 0, 0), thickness=2)
cv2.rectangle(f, (x,y), (x+w,y+h), color=(255, 0, 255), thickness=2)
cv2.imshow('boundingRect', f)


# Momentos del contorno
M=cv2.moments(cnt)
huMoments = cv2.HuMoments(M)  # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gab001db45c1f1af6cbdbe64df04c4e944


# --- Hough Lineas --------------------------------------------------------------------------------
# Tutorial: 
#   https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
#   https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
f = cv2.imread('contornos.png')             # Leemos imagen
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)  # Pasamos a escala de grises
imshow(gray)

edges = cv2.Canny(gray, 100, 170, apertureSize=3)
cv2.imshow('imagen', edges)

lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=250)   # https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
# lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)   # https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]        
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*(a))
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*(a))
    cv2.line(f,(x1,y1),(x2,y2),(0,255,0),2)

# cv2.imshow('hough lines', f)
imshow(f)


# --- Hough Circulos --------------------------------------------------------------------------------
# Tutorial: https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
# https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
img = cv2.imread('logo_opencv.png', cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(img,5)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=40, minRadius=0, maxRadius=50)  # Circulos chicos
# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=30, minRadius=0, maxRadius=50) # Circulos grandes + otros
# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=50, minRadius=0, maxRadius=50) # Circulos grandes
circles = np.uint16(np.around(circles))
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for i in circles[0,:]:
    cv2.circle(cimg, (i[0],i[1]), i[2], (0,255,0), 2)   # draw the outer circle
    cv2.circle(cimg, (i[0],i[1]), 2, (0,0,255), 2)      # draw the center of the circle
imshow(cimg)


# -------------------------------------------------------------------------------
# --- Umbralado -----------------------------------------------------------------
# -------------------------------------------------------------------------------
img = cv2.imread("text.png",cv2.IMREAD_GRAYSCALE)
imshow(img)

# --- Manual -------------------------------------------
T = (img.min() + img.max())/2
flag = False
while ~flag:
   g = img >= T
   Tnext = 0.5*(np.mean(img[g]) + np.mean(img[~g]))
   flag = np.abs(T-Tnext) < 0.5
   T = Tnext
print(T)
img_th_manual = img>T
_, img_th_manual = cv2.threshold(img, thresh=T, maxval=255, type=cv2.THRESH_BINARY)

 # --- Otsu --------------------------------------------
T_otsu, img_th_otsu = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_OTSU)

# --- Graficas -----------------------------------------
plt.figure()
ax1 = plt.subplot(221)
plt.xticks([]), plt.yticks([]), plt.imshow(img, cmap="gray"), plt.title('Imagen Original')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(~img_th_manual, cmap="gray"), plt.title(f'Umbral Manual {T:5.2f}')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(~img_th_otsu, cmap="gray"), plt.title(f'Umbral Otsu {T_otsu:5.2f}')
plt.show(block=False)


