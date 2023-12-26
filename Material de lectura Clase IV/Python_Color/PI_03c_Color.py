import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image   # Pillow --> https://pillow.readthedocs.io/en/stable/

# --- Imagen RGB ------------------------------------------------------------------------
img = cv2.imread('peppers.png')
plt.figure(1)
plt.imshow(img) # Acá se puede observar que OpenCV carga la imgagen como BGR.
plt.show()

# --- Acomodamos canales ----------------------------------------------------------------
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(2)
plt.imshow(img_RGB)
plt.show()

# --- Separar canales -------------------------------------------------------------------
B, G, R = cv2.split(img)
plt.figure(3)
plt.imshow(R, cmap='gray')
plt.title("Canal R")
plt.show()

ax1 = plt.subplot(221)
plt.xticks([]), plt.yticks([]), plt.imshow(img_RGB), plt.title('Imagen RGB')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(R,cmap='gray'), plt.title('Canal R')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(G,cmap='gray'), plt.title('Canal G')
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(B,cmap='gray'), plt.title('Canal B')
plt.show()

# --- Modifico un canal ---------------------------------------------------------------
# img2 = img_RGB    # No! así crea una referencia: si se modifica una, se modifica la otra también.  
img2 = img_RGB.copy()  # Así crea una copia. Otra forma sería "img2 = np.array(img_RGB)"
img2[:,:,0] = 0
plt.figure, plt.imshow(img2), plt.title('Canal R anulado')
plt.show()

R2 = R.copy()
R2 = R2*0.5
R2 = R2.astype(np.uint8)
img3 = cv2.merge((R2,G,B))
plt.figure, plt.imshow(img3), plt.title('Canal R escalado')
plt.show()

ax1 = plt.subplot(221)
plt.xticks([]), plt.yticks([]), plt.imshow(img_RGB), plt.title('Imagen RGB')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img2,cmap='gray'), plt.title('Canal R anulado')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img3), plt.title('Canal R escalado')
plt.show()


# --- Imagen Indexada -------------------------------------------------------------------
img = cv2.imread('peppers.png')       #  Nbytes_img_idx/Nbytes_img = 1.83  
# img = cv2.imread('home.jpg')        #  Nbytes_img_idx/Nbytes_img = 1.59
# img = cv2.imread('flowers.tif')     #  Nbytes_img_idx/Nbytes_img = 1.667  # Cuidado! puede demorar mucho en correr...
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Obtengo colores
img_pixels = img.reshape(-1,3)
# colours = np.unique(img_pixels, axis=0)
colours, counts = np.unique(img_pixels, axis=0, return_counts=True)
idx = np.argsort(counts)        # Opcional:
counts = counts[idx]            # Ordeno los colores segun su frecuencia de aparición
colours = colours[idx]          # El ultimo elemento de counts posee el color con mayor frec. de aparición.
N_colours = colours.shape[0]

# Genero imagen indexada
img_idx = -np.ones(img.shape[:-1])
for ii in range(N_colours):
    # # --- Version legible ---------------------------------------------------------------------
    # col_sel = colours[ii]
    # maskR = img[:,:,0] == col_sel[0]
    # maskG = img[:,:,1] == col_sel[1]
    # maskB = img[:,:,2] == col_sel[2]
    # mask = maskR & maskG & maskB
    # img_idx[mask] = ii
    # # --- Version compacta -------------------------------------------------------------------
    # img_idx[(img[:,:,0] == colours[ii][0]) & (img[:,:,1] == colours[ii][1]) & (img[:,:,2] == colours[ii][2])] = ii
    # --- Otra version -------------------------------------------------------------------------
    mask = cv2.inRange(img, colours[ii], colours[ii])  # https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981
    img_idx[mask>0] = ii


# Check & Conversion
np.any(img_idx==-1) # Verificamos que ningún pixel quedó sin asignar...
img_idx = np.int32(img_idx)
img_idx.dtype

# Calculo de relación de bytes
Nbytes_img_idx = 4*np.prod(img_idx.shape) + np.prod(colours.shape)
Nbytes_img = np.prod(img.shape)
Nbytes_img_idx/Nbytes_img

# Plots
plt.figure()
ax1 = plt.subplot(221)
plt.xticks([]), plt.yticks([]), plt.imshow(img), plt.title('Imagen Original'), plt.colorbar()
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img_idx, cmap="gray"), plt.title('Imagen Indexada - Indices'), plt.colorbar()
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img_idx, cmap="jet"), plt.title('Imagen Indexada - jet'), plt.colorbar()
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(img_idx, cmap="hot"), plt.title('Imagen Indexada - hot'), plt.colorbar()
plt.show(block=False)

# Plot opcional (cuando los colores están ordenados)
P = 95 # 95 - 80 - 50 -20
img_idx_topValues_mask = cv2.inRange(img_idx, (N_colours-1)*P/100, (N_colours-1))
img_idx_topValues = cv2.bitwise_and(img, img, mask= img_idx_topValues_mask)
# img_idx_topValues = img.copy()                            # Lo mismo que antes...   
# img_idx_topValues[~(img_idx_topValues_mask>0),:] = 0      # pero de manera "manual".
porc = 100* np.sum(img_idx_topValues_mask>0) / np.prod(img_idx_topValues_mask.shape)

plt.figure()
ax1 = plt.subplot(221)
plt.xticks([]), plt.yticks([]), plt.imshow(img), plt.title('Imagen Original'), plt.colorbar()
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img_idx_topValues_mask, cmap="gray"), plt.title(f'Imagen Indexada - {P:5.2f}% top values - mask'), plt.colorbar()
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img_idx_topValues), plt.title(f'Imagen Indexada - top values (%{porc:5.2f})'), plt.colorbar()
plt.show(block=False)


# --- Dithering ------------------------------------------------------------------------
img_PIL = Image.open('cameraman.tif')
image_dithering = img_PIL.convert(mode='1', dither=Image.FLOYDSTEINBERG)   # https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=convert#PIL.Image.Image.convert
plt.figure(4)
ax1 = plt.subplot(121)
plt.xticks([]), plt.yticks([])
plt.imshow(img_PIL, cmap='gray'), plt.title('Imagen original')
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(image_dithering), plt.title('Imagen con dithering')
plt.show()
# -----------------------------------
# --- Analisis ----------------------
# -----------------------------------
# -- Imagen Original --------
img_PIL.size
x = img_PIL.getdata()
list_of_pixels = list(x)
len(list_of_pixels)
list_of_pixels[:5]
# print(list_of_pixels)
Ncolors = len(list(set(list_of_pixels)))
# -- Imagen Procesada --------
list_of_pixels_out = list(image_dithering.getdata())
len(list_of_pixels_out)
list_of_pixels_out[:5]
Ncolors_out = len(list(set(list_of_pixels_out)))
# -----------------------------------
# -----------------------------------
# -----------------------------------


# Color - Ejemplo 1
img_PIL = Image.open('landscape.jpg')
img_proc = img_PIL.convert(mode="P", dither=Image.NONE, palette=Image.WEB)  # standard 216-color "web palette"
img_proc_dither = img_PIL.convert(mode="P", dither=Image.FLOYDSTEINBERG, palette=Image.WEB)  # standard 216-color "web palette"
plt.figure()
ax1 = plt.subplot(221)
plt.xticks([]), plt.yticks([]), plt.imshow(img_PIL), plt.title('Imagen Original')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img_proc), plt.title('Imagen procesada')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img_proc_dither), plt.title('Imagen procesada + dither')
plt.show()



# Color - Ejemplo 2
img_PIL = Image.open('peppers.png')
image_dithering = img_PIL.convert(mode='P', palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG)  # https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
# image_dithering = img_PIL.convert(mode='P', palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG, colors=3)
image_dithering = img_PIL.convert(mode='P', palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG, colors=8)
plt.figure(5)
ax1 = plt.subplot(121)
plt.xticks([]), plt.yticks([])
plt.imshow(img_PIL), plt.title('Imagen original')
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(image_dithering), plt.title('Imagen con dithering')
plt.show(block=False)
# -----------------------------------
# --- Analisis ----------------------
# -----------------------------------
# -- Imagen Original --------
img_PIL.size
x = img_PIL.getdata()
list_of_pixels = list(x)
len(list_of_pixels)
list_of_pixels[:5]
# print(list_of_pixels)
Ncolors = len(list(set(list_of_pixels)))
# -- Imagen Procesada --------
list_of_pixels_out = list(image_dithering.getdata())
len(list_of_pixels_out)
list_of_pixels_out[:5]
Ncolors_out = len(list(set(list_of_pixels_out)))

image_dithering.getcolors() # [ ( count, index ), ( count, index ), ... ]
palette = np.array(image_dithering.getpalette(),dtype=np.uint8).reshape((256,3))
palette[0:4,]

# Paso a RGB
image_dithering_RGB = np.array(image_dithering.convert('RGB'))  # Paso a RGB
colours, counts = np.unique(image_dithering_RGB.reshape(-1,3), axis=0, return_counts=1)    # Obtengo colores y cuentas

# Grafico de torta --> Tener cuidado con la cantidad de colores! debe ser relativamente chica para hacer este grafico
idx = np.argsort(-counts)   # Esto es opcional...
counts = counts[idx]        # Ordeno en base a frecuencia de ocurrencias 
colours = colours[idx]      # 
counts_pct = counts/np.sum(counts)*100  # Paso a porcentaje
# labels = [f'({c[0]},{c[1]},{c[2]})' for c in colours]
# labels = [f'{counts_pct[ii]:5.2f}%  ({colours[ii,0]},{colours[ii,1]},{colours[ii,2]})' for ii in range(len(counts))]
labels = [f'{counts_pct[ii]:6.2f}%  ({colours[ii,0]:3d},{colours[ii,1]:3d},{colours[ii,2]:3d})' for ii in range(len(counts))]
col = [(c[0]/255., c[1]/255., c[2]/255.) for c in colours]

fig= plt.figure(figsize=(9,5))
ax = fig.add_subplot(111) 
ax.pie(counts_pct, labels=labels, colors=col)
pos1 = ax.get_position()
pos2 = [0.15, pos1.y0, pos1.width, pos1.height] 
ax.set_position(pos2) 
plt.legend(title = "Colores RGB:", bbox_to_anchor=(1.4, 1), loc='upper left', borderaxespad=0.5, fontsize=10)   # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
plt.title("Proporción de Colores")
plt.show(block=False)

# Obtengo índices
image_dithering_indexs = np.array(image_dithering.convert('L'))  # Matriz de índices
indexs, counts = np.unique(image_dithering_indexs, return_counts=1)
plt.figure, plt.imshow(image_dithering_indexs, cmap='gray'), plt.colorbar(), plt.show()
# -----------------------------------
# -----------------------------------
# -----------------------------------

# --- Espacio de color HSV ----------------------------------------------
img = cv2.imread('flowers.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Rangos --> H: 0-179  / S: 0-255  / V: 0-255
h, s, v = cv2.split(img_hsv)
plt.figure()
plt.subplot(221), plt.imshow(img)
plt.subplot(222), plt.imshow(h, cmap='gray'), plt.title('Canal H')
plt.subplot(223), plt.imshow(s, cmap='gray'), plt.title('Canal S')
plt.subplot(224), plt.imshow(v, cmap='gray'), plt.title('Canal V')
plt.show()

# --- Espacio de color HSI ----------------------------------------------
img = cv2.imread('flowers.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) 
h, l, s = cv2.split(img_hls)
plt.figure()
plt.subplot(221), plt.imshow(img)
plt.subplot(222), plt.imshow(h, cmap='gray'), plt.title('Canal H')
plt.subplot(223), plt.imshow(l, cmap='gray'), plt.title('Canal L')
plt.subplot(224), plt.imshow(s, cmap='gray'), plt.title('Canal S')
plt.show()

h_smooth = cv2.blur(h, (9, 9))
l_smooth = cv2.blur(l, (9, 9))
s_smooth = cv2.blur(s, (9, 9))
img_smooth_all = cv2.cvtColor(cv2.merge((h_smooth, l_smooth, s_smooth)), cv2.COLOR_HLS2RGB)
img_smooth_l = cv2.cvtColor(cv2.merge((h, l_smooth, s)), cv2.COLOR_HLS2RGB)

plt.figure()
ax1=plt.subplot(221) 
plt.imshow(img), plt.title('Imagen Original')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img_smooth_l, cmap='gray'), plt.title('Blur en canal L')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img_smooth_all, cmap='gray'), plt.title('Blur en los 3 canales')
plt.show()


# --- Espacio de color HSV - Ejemplo ----------------------------------------------
img = cv2.imread('peppers.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Rangos --> H: 0-179  / S: 0-255  / V: 0-255
h, s, v = cv2.split(img_hsv)
plt.figure()
plt.subplot(221), plt.imshow(img)
plt.subplot(222), plt.imshow(h, cmap='gray'), plt.title('Canal H')
plt.subplot(223), plt.imshow(s, cmap='gray'), plt.title('Canal S')
plt.subplot(224), plt.imshow(v, cmap='gray'), plt.title('Canal V')
plt.show()

# Segmentacion en color - Detectar solo el rojo
ix_h1 = np.logical_and(h > 180 * .9, h < 180)
ix_h2 = h < 180 * 0.04
ix_s = np.logical_and(s > 255 * 0.3, s < 255)
ix = np.logical_and(np.logical_or(ix_h1, ix_h2), ix_s)
# ix2 = (ix_h1 | ix_h2) & ix_s   # Otra opcion que da igual...

r, g, b = cv2.split(img)
r[ix != True] = 0
g[ix != True] = 0
b[ix != True] = 0
rojo_img = cv2.merge((r, g, b))
plt.figure(7)
plt.imshow(rojo_img)
plt.show()

# --- Filtrado espacial ----------------------------------------------------------------
img = cv2.imread('peppers.png')
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Usando kernel y filter 2D
kernel = np.ones((5, 5), np.float32)/25
img_filt = cv2.filter2D(img_RGB, -1, kernel)
plt.figure(8)
plt.imshow(img_filt)
plt.show()

# Funciones filtrado
gblur = cv2.GaussianBlur(img_RGB, (55, 55), 0)
median = cv2.medianBlur(img_RGB, 21)
blur = cv2.blur(img_RGB, (55, 55))
plt.figure(9)
plt.subplot(221), plt.imshow(img_RGB), plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(blur), plt.title('Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(gblur), plt.title('Gaussian blur'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(median), plt.title('Median blur'), plt.xticks([]), plt.yticks([])
plt.show()

# Filtrado Espacial - High Boost
img = cv2.imread('flowers.tif')
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
w1 = np.ones((3, 3), np.float32)/9
w2 = np.ones((3, 3), np.float32)  # Laplaciano  
w2[1,1] = -8                      #
# -----------------
def im2double(im):
    info = np.iinfo(im.dtype) 
    return im.astype(np.float) / info.max 

img_RGB = im2double(img_RGB)
# ------------------
img_pb = cv2.filter2D(img_RGB, -1, w1)
img_en = img_pb - cv2.filter2D(img_pb, -1, w2)
plt.figure(10)
ax1 = plt.subplot(221)
plt.imshow(img_RGB)
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(img_pb), plt.title('Filtro Pasa-Bajos en todos los canales'), plt.xticks([]), plt.yticks([])
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(img_en), plt.title('Mejorada utilizando Laplaciano'), plt.xticks([]), plt.yticks([])
plt.show()

