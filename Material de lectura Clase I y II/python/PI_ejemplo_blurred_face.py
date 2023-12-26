import  cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("MonaLisa.jpg")
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

img2 = img.copy()
img3 = img.copy()
x = 100
y = 480
w = 100
h = 75
cv2.rectangle(img2, (y,x), (y+h,x+w), (0,255,255), 2)   # Agrego un rectángulo  (https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html?highlight=rectangle)

sub_face = img[x:x+w, y:y+h]
sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)      # Aplico borrosidad...
img3[x:x+w, y:y+h] = sub_face                           # ... y reemplazo en la imagen

plt.figure()
plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
plt.figure()
plt.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB))
plt.show()


# --- Version con máscaras -------------------------------------
mask = np.zeros(img.shape,np.uint8)
mask[x:x+w, y:y+h] = 1

maskneg = mask.copy()
maskneg[maskneg==1] = 255
maskneg[maskneg==0] = 1
maskneg[maskneg==255] = 0

plt.subplot(121)
plt.imshow(mask[:,:,0],cmap='gray')
plt.subplot(122)
plt.imshow(maskneg[:,:,0],cmap='gray')
plt.show()

img_blurred = cv2.GaussianBlur(img,(23, 23), 30)
img_face = img_blurred*mask
img_back = img*maskneg

plt.subplot(121)
plt.imshow(cv2.cvtColor(img_face,cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(cv2.cvtColor(img_back,cv2.COLOR_BGR2RGB))
plt.show()

img_out = img_face + img_back
plt.imshow(cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB))
plt.show()

