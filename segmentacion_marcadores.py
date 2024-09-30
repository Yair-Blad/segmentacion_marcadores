import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('catedral.jpg')
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gris, (5, 5), 0)

gradiente_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
gradiente_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
magnitud = cv2.magnitude(gradiente_x, gradiente_y)
_, bordes = cv2.threshold(magnitud, 50, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
bordes = cv2.dilate(bordes, kernel, iterations=2)
bordes = cv2.erode(bordes, kernel, iterations=1)
bordes = cv2.convertScaleAbs(bordes)

ret, marcadores = cv2.connectedComponents(bordes)
marcadores = marcadores + 1

bordes_inv = cv2.bitwise_not(bordes)
marcadores[bordes_inv == 255] = 0

imagen_watershed = img.copy()
cv2.watershed(imagen_watershed, marcadores)
imagen_watershed[marcadores == -1] = [255, 0, 0]

plt.figure(figsize =(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Imagen original: ')

plt.subplot(1, 3, 2)
plt.imshow(bordes, cmap='gray')
plt.title('Bordes detectados: ')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(imagen_watershed, cv2.COLOR_BGR2RGB))
plt.title('Segmentacion con marcadores:')

plt.show()



