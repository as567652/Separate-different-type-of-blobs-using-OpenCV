import cv2 as cv
import numpy as np
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
import random as rng

original_img = cv.imread('main/images/image_a5_blobs.tif')
im = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)

plt.figure()
plt.title('Original Image')
plt.imshow(original_img, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.show()

ret, bw_img = cv.threshold(im, 100, 255, cv.THRESH_BINARY)

canny_output = cv.Canny(bw_img,127,200)

plt.figure()
plt.title('Canny Edge Detection')
plt.imshow(canny_output, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.show()

contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)
centers = [None]*len(contours)
radius = [None]*len(contours)
for i, c in enumerate(contours):
    contours_poly[i] = cv.approxPolyDP(c, 3, True)
    centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    
rad_dic = {}

for i in range(len(contours)):
    r = int(radius[i])
    if r in rad_dic:
        rad_dic[r]+=1
    else:
        rad_dic[r]=1

all_availaibe_radius = list(rad_dic)

#print(all_availaibe_radius)

fig = plt.figure(figsize=(10, 10))
rows = 2
columns = 2

cnt = 1
for rad in all_availaibe_radius:
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    drawing[:] = 255
    threshold = 1
    for i in range(len(contours)):
        if int(radius[i]) == rad and rad_dic[rad] > threshold:
            color = (76, 76, 76)
            #cv.drawContours(drawing, contours_poly, i, color)
            cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, -1)
    if rad_dic[rad] > threshold:
        fig.add_subplot(rows, columns, cnt)
        plt.imshow(drawing)
        plt.axis('off')
        plt.title("Blobs Of Radius " + str(rad))
        cnt+=1
plt.show()