import cv2
import numpy as np

# CONSTANTES
C_RANGE = np.linspace(5/9, 1, 400)
C_RANGE = C_RANGE[::-1]
D_RANGE = np.linspace(0.5, 5, 400)

def update(_):
    # --- Obtengo parámetros de la GUI -----------------------------
    _desv = cv2.getTrackbarPos("Desviación Standard", "Filtrado Unsharp")
    _peso = cv2.getTrackbarPos("Peso", "Filtrado Unsharp")
    # --- Proceso --------------------------------------------------
    # ===========================================================
    # Filtro Unsharp:
    #
    #    F = c/(2c-1) * I  +  (1-c)/(2c-1) * U
    #   
    # donde U es la imagen I suavizada con filtro Gaussiano.
    #
    # Parámetros:
    #   w:      Tamaño del kernel (Filtro Gaussiano) 
    #   sigma:  Desviación std (Filtro Gaussiano) 
    #   c:      Peso en la combinación
    # =========================================================== 
    w = 9
    desv = D_RANGE[_desv]
    C = C_RANGE[_peso]
    print(f"Desvio: {desv:6.3f} ({_desv})  -  Peso: {C:6.3f} ({_peso})   -  kernel_size: {w}")
    img_blur = cv2.GaussianBlur(source_img, ksize=(w, w), sigmaX=desv, sigmaY=desv)
    img_modif = (C/(2*C-1))*source_img - ((1-C)/(2*C-1))*img_blur
    # --- Actualizo ------------------------------------------------
    cv2.imshow("Filtrado Unsharp", img_modif)
    cv2.resizeWindow("Filtrado Unsharp", 675, img_modif.shape[1])
       
# --- Imagen de entrada ----------------------------------------------------
src_path = "Fish_Scale_BF.png"
# src_path = "Pine_Cone_Tissue.png"
source_img = cv2.imread(src_path, cv2.IMREAD_COLOR)
source_img = np.float32(source_img)/255.0   # Conversión a float
cv2.imshow("Imagen Original", source_img)

# --- GUI ------------------------------------------------------------------
cv2.namedWindow("Filtrado Unsharp",cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("Desviación Standard", "Filtrado Unsharp", 0, 399, update)
cv2.createTrackbar("Peso", "Filtrado Unsharp", 0, 399, update)

# --- Loop hasta presionar ESC. -------------------------------------------
update(None)
while True:
    ch = cv2.waitKey()
    if ch == 27:
        break
    if ch == ord(" "):
        defocus = not defocus
        update(None)
