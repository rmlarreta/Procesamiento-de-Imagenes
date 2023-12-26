import cv2
import numpy as np
import matplotlib.pyplot as plt

def imclearborder(f):
    kernel = np.ones((3,3),np.uint8)
    marker = f.copy()
    marker[1:-1,1:-1]=0
    while True:
        tmp=marker.copy()
        marker=cv2.dilate(marker, kernel)
        marker=cv2.min(f, marker)
        difference = cv2.subtract(marker, tmp)
        if cv2.countNonZero(difference) == 0:
            break
    mask=cv2.bitwise_not(marker)
    out=cv2.bitwise_and(f, mask)
    return out    


# f = cv2.imread('5R0Zs.jpg', cv2.IMREAD_GRAYSCALE)
f = cv2.imread('book_text_bw.tif', cv2.IMREAD_GRAYSCALE)
thresh = cv2.threshold(f, 40, 255, cv2.THRESH_BINARY)[1]

f_clearb = imclearborder(thresh)

plt.figure()
ax1 = plt.subplot(121)
plt.imshow(f, cmap='gray')
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(f_clearb, cmap='gray'), plt.title('Bordes Limpios'), plt.xticks([]), plt.yticks([])
plt.show()

