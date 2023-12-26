import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('xray-chest.png',cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray') 
plt.show()