import numpy as np
import cv2

TOL_REPOSO = 0.03 # tolerancia para detectar movimiento de los dados
LOW_COLOR = (0, 75, 130) # nivel de rojo inferior para segmentación usando CIELAB
HIGH_COLOR = (255, 123, 150) # nivel de rojo superior para segmentación usando CIELAB
LOW_AREA = 4000 # umbral inf. de área para indentificar un dado
HIGH_AREA = 6000 # umbral sup. de área para indentificar un dado
LOW_ASPECT_RATIO = 0.85 # límite inf. de alargamiento para indentificar un dado
HIGH_ASPECT_RATIO = 1.15 # límite sup. de alargamiento para indentificar un dado
MEDIAN_BLUR = 7 # tamaño del filtro utilizado
THRESHOLD = 170 # threshold utilizado para el umbralado
CLOSE_ELEMENT = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # Elemento estructural utilizado al aplicar clausura
FRAME_STEP = 3 # Cada cuantos frames se actualizan los datos del frame anterior para detectar mov.

videos = ["tirada_1.mp4", "tirada_2.mp4", "tirada_3.mp4", "tirada_4.mp4"]

for f in videos:

    cap = cv2.VideoCapture(f)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    f_split = f.split(".", 2)
    f_out = f_split[0] + "_out_v2." + f_split[1]

    # cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f_out, -1, fps, (width, height))

    # ls_dados y ls_dados_ant son listas de diccionarios, donde para cada dado se guarda
    # - El objeto contorno del dado
    # - Un contador con la cantidad de puntos del dado.

    # ls_dados_ant guarda el estado de ls_dados para comparar en la prox. iteración
    ls_dados_ant = None
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            # --- Procesamiento ---------------------------------------------
            mask_dados = np.zeros((height, width, 3), dtype=np.uint8)

            frame_out = frame.copy() # Frame del video de salida (Img. Derecha)

            # Nro de frame
            id = cap.get(1)
            cv2.putText(frame_out, "Frame: {:d}".format(int(id)), org=(
                100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            # Filtrado por color
            # PI_03c_Color_Segmentacion_CIELAB_video.py
            # filtrado_cielab.py
            frame_LAB = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            frame_threshold = cv2.inRange(frame_LAB, LOW_COLOR, HIGH_COLOR)
            
            #Contornos
            contours, hierarchy = cv2.findContours(
                frame_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            count_dados = 0
            ls_dados = []
            for ii, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                aspect_ratio = float(w)/h

                # Descriptores geométricos - Area y Relación de Aspecto (Alargamiento)
                if (area > LOW_AREA and area < HIGH_AREA and aspect_ratio >= LOW_ASPECT_RATIO and aspect_ratio <= HIGH_ASPECT_RATIO):
                    _ = cv2.rectangle(
                        frame_out, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)
                    # Guardo el contorno relleno del dado en la máscara mask_dados
                    _ = cv2.drawContours(mask_dados, contours, contourIdx=ii, color=(
                        255, 255, 255), thickness=cv2.FILLED)
                    ls_dados.append({"cnt": cnt, "value": 0})
                    count_dados += 1

            # Determino el reposo en función de la cantidad de dados que tienen una posición (x1, y1)
            # cercana a algún dado de un frame anterior (guardado en ls_dados_ant)
            reposo = False
            if not ls_dados_ant is None:

                count_dados_reposo = 0
                if count_dados == 5:
                    
                    for dado_ant in ls_dados_ant:
                        for dado in ls_dados:
                            x1, y1, w1, h1 = cv2.boundingRect(dado["cnt"])
                            x2, y2, w2, h2 = cv2.boundingRect(dado_ant["cnt"])
                            if x1*(1-TOL_REPOSO) <= x2 <= x1*(1+TOL_REPOSO) and y1*(1-TOL_REPOSO) <= y2 <= y1*(1+TOL_REPOSO):
                                count_dados_reposo += 1

                    cv2.putText(frame_out, "Dados en reposo: " + str(count_dados_reposo) , org=(100, 135), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                else:
                    cv2.putText(frame_out, "Dados en reposo: -", org=(100, 135), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

                # Si la cantidad de dados es 5 implica reposo
                if count_dados == count_dados_reposo == 5:
                    reposo = True
                    cv2.rectangle(frame_out, (5, 5),
                                  (width-5, height-5), (0, 255, 255), 5)

            # frame_pre: Frame del video con resultado del procesamiento intermedio (Img. Izquierda)

            # Determinar valores de los dados
            if reposo:
                # Tomo los dados de la imagen original a partir de la máscara mask_dados.    
                dados = cv2.bitwise_and(frame, frame, mask=cv2.cvtColor(
                    mask_dados, cv2.COLOR_BGR2GRAY))
                frame_pre = cv2.cvtColor(dados, cv2.COLOR_BGR2GRAY)

                # Filtro de Mediana
                frame_pre = cv2.medianBlur(frame_pre, MEDIAN_BLUR)
                # Umbralado
                umbral, frame_pre = cv2.threshold(
                    frame_pre, thresh=THRESHOLD, maxval=255, type=cv2.THRESH_BINARY)
                # Clausura
                frame_pre = cv2.morphologyEx(frame_pre, cv2.MORPH_CLOSE, CLOSE_ELEMENT)

                # Contornos
                # hierarchy: [Next, Previous, First_Child, Parent]
                contours, _ = cv2.findContours(
                    frame_pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                # Recorro puntos de los dados
                for ii, cnt in enumerate(contours):
                    _ = cv2.drawContours(frame_out, contours, contourIdx=ii, color=(0, 255, 0), thickness=2)
                    try:
                        # Calculo el centroide de cada punto para determinar en que dado está contenido
                        M = cv2.moments(cnt)
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    except:
                        pass
                    else:
                        for dado in ls_dados:
                            x, y, w, h = cv2.boundingRect(dado["cnt"]) 
                            if x <= cX <= x+w and y <= cY <= y+h:
                                dado["value"] += 1
                                break

                for dado in ls_dados:
                    x, y, w, h = cv2.boundingRect(dado["cnt"])
                    cv2.putText(frame_out, str(dado["value"]), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

                frame_pre = cv2.cvtColor(frame_pre, cv2.COLOR_GRAY2BGR)
            else:
                frame_pre = mask_dados

            # --- Visualización ---------------------------------------------
            
            frame_resized_pre = cv2.resize(frame_pre, dsize=(
                int(width / 3), int(height / 3)))
            frame_resized_out = cv2.resize(frame_out, dsize=(
                int(width / 3), int(height / 3)))

            cv2.imshow(f, np.concatenate(
                (frame_resized_pre, frame_resized_out), axis=1))

            # --- Salida ----------------------------------------------------
            # grabo frame
            out.write(frame_out)

            # Se alamacena cada n frames los datos del frame actual para poder
            # detectar movimiento
            if id % FRAME_STEP == 0:
                ls_dados_ant = ls_dados.copy()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
