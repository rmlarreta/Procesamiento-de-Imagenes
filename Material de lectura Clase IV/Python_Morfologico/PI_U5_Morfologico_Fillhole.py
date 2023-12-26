import cv2
import numpy as np
import matplotlib.pyplot as plt

def fillhole(input_image):
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)  # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#floodfill
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out 


f = cv2.imread('book_text_bw.tif', cv2.IMREAD_GRAYSCALE)
_,fth = cv2.threshold(f,127,255,cv2.THRESH_BINARY)
f_fill = fillhole(fth)

plt.figure()
ax1 = plt.subplot(121)
plt.imshow(f, cmap='gray')
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122,sharex=ax1,sharey=ax1), plt.imshow(f_fill, cmap='gray'), plt.title('Rellenado de huecos'), plt.xticks([]), plt.yticks([])
plt.show()
