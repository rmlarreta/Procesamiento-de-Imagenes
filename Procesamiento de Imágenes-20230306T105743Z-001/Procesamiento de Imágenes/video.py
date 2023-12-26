import numpy as np
import cv2

videos = ["tirada_1.mp4", "tirada_2.mp4", "tirada_3.mp4", "tirada_4.mp4"]

for f in videos:

    cap = cv2.VideoCapture(f)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    f_split = f.split(".", 2)
    f_out = f_split[0] + "_out." + f_split[1]
    
    #cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f_out, -1, fps, (width, height))
    
    # Máscara de los dados del n frame anterior
    mask_dados_ant = None 
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            # --- Procesamiento ---------------------------------------------
            mask_dados = np.zeros((height, width, 3), dtype=np.uint8)
            
            frame_out = frame.copy()
            
            # Nro de frame
            id = cap.get(1)
            cv2.putText(frame_out, "Frame: {:d}".format(int(id)), org=(
                100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

            # Filtrado por color
            # PI_03c_Color_Segmentacion_CIELAB_video.py
            # filtrado_cielab.py
            low_color = (0, 75, 130)
            high_color = (255, 123, 150)
            frame_LAB = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            frame_threshold = cv2.inRange(frame_LAB, low_color, high_color)
            
            # Contornos    
            contours, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            count_dados = 0
            for ii, cnt in enumerate(contours):
                x,y,w,h = cv2.boundingRect(contours[ii])
                area = cv2.contourArea(cnt)
                aspect_ratio = float(w)/h
                leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
                rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
                topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
                bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

                # Descriptores geométricos - Area y Relación de Aspecto (Alargamiento)
                if (area > 4000 and area < 6000 and aspect_ratio>=0.85 and aspect_ratio<=1.15):
                    _ = cv2.rectangle(frame_out, (x,y), (x+w,y+h), color=(255, 0, 0), thickness=2)
                    _ = cv2.drawContours(mask_dados, contours, contourIdx=ii, color=(255, 255, 255), thickness=cv2.FILLED)
                    count_dados += 1
    
            # RMSE mask_dados actual vs mask_dados_ant (máscara anterior)
            # Si tengo un frame previo, comparo con el actual via RMSE y determino si la img. está en reposo.
            # No funciona bien con la imagen original. Cambios de ilumanción / enfoque alteran esto.
            
            reposo = False
            if not mask_dados_ant is None:

                # Calculo el rmse sólo si hay 5 dados
                if count_dados==5:  
                    rmse = np.sqrt(np.mean((mask_dados_ant.astype(float) - mask_dados.astype(float)) ** 2))
                    cv2.putText(frame_out, "RMSE: {:f}".format(rmse), org=(100, 135), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)    
                else:
                    cv2.putText(frame_out, "RMSE: -", org=(100, 135), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)       
                
                # Si la cantidad de dados es 5 y el RMSE es menor a 10, implica reposo
                if count_dados==5 and rmse < 10:
                    reposo = True
                    cv2.rectangle(frame_out, (5,5), (width-5,height-5), (0,255,255), 5) 

            #Determinar valores de los dados
            if reposo:
      
                dados = cv2.bitwise_and(frame, frame, mask=cv2.cvtColor(mask_dados, cv2.COLOR_BGR2GRAY))
                frame_pre = cv2.bitwise_or(dados, ~mask_dados)
                frame_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY) 
                
                # Filtro de Mediana
                frame_pre = cv2.medianBlur(frame_pre, 7)
                # # Umbralado
                umbral, frame_pre = cv2.threshold(frame_pre, thresh=150, maxval=255, type=cv2.THRESH_BINARY)
                
                # Clausura
                frame_pre = cv2.morphologyEx(frame_pre, cv2.MORPH_CLOSE, 
                    cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
                
                # Contornos
                # hierarchy: [Next, Previous, First_Child, Parent]
                contours, hierarchy = cv2.findContours(frame_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                
                for ii, cnt in enumerate(contours):
                    
                    # Contornos nivel 0 - hijos del contorno principal
                    if hierarchy[0][ii][3]==0:

                        # Cantidad contornos hijos del contorno
                        hijos = 0
                        for jj in range(len(contours)):
                            if hierarchy[0][jj][3]==ii:
                                hijos += 1
                        
                        x,y,w,h = cv2.boundingRect(contours[ii])
                     
                        cv2.putText(frame_out, str(hijos), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)   

                frame_pre = cv2.cvtColor(frame_pre, cv2.COLOR_GRAY2BGR)
            else:
                frame_pre = mask_dados

            # --- Visualización ---------------------------------------------
            frame_resized_pre = cv2.resize(frame_pre, dsize=(
                int(width / 3), int(height / 3)))
            frame_resized_out = cv2.resize(frame_out, dsize=(
                int(width / 3), int(height / 3)))
            
            cv2.imshow(f, np.concatenate((frame_resized_pre, frame_resized_out), axis=1))

            # --- Salida ----------------------------------------------------
            # grabo frame
            out.write(frame_out)  

            # Se alamacena el frame actual para calcular la diferencia con el próximo frame
            # La actualización de esta variable ocurre cada n frames (en el código 2)
            if id % 2 == 0:
                 mask_dados_ant = mask_dados.copy()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
