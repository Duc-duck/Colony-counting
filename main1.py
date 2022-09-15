from math import floor
from this import s
import cv2
import pandas as pd
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from collections import Counter
import matplotlib.pyplot as plt
import sys
from PIL import Image, ImageDraw
np.set_printoptions(threshold = sys.maxsize)

def detectAndTrimDisk(image, padding_size):
    
    minRadius = 0
    maxRadius = image.shape[1] #image width
    minDist = 100

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=minDist,minRadius=minRadius, maxRadius=maxRadius)

    biggestRadius = 0 
    biggestCenter = (0,0)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            if i[2] > biggestRadius:
                biggestRadius = i[2]
                biggestCenter = (i[0], i[1])
                
    mask = np.zeros_like(image)
    
    cv2.circle(mask, biggestCenter, biggestRadius - padding_size, (255, 255,255), -1)
    #cv2.imshow('image', image)
    roi = cv2.bitwise_and(image, mask)
    roi[roi <= 10] = 255
    return roi

def draw_function(image, x, y):
    global b, g, r, x_pos, y_pos
    x_pos = x
    y_pos = y
    b, g, r = image[y, x]
    b = int(b)
    g = int(g)
    r = int(r)
    return b, g, r

def get_color_name(R, G, B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

r = g = b = x_pos = y_pos = 0
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv(r'C:\Users\Victus\Desktop\Color-Detection-OpenCV-main\colors.csv', names=index, header=None)

image = cv2.imread(r"C:\Users\Victus\Desktop\Color-Detection-OpenCV-main\1.jpg")

image = detectAndTrimDisk(image, 20)
cv2.imshow('Cropped Image', image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
# with  open(r"C:\Users\Victus\Desktop\file1.txt", "w") as file:
#  	file.write(str(gray[100:300, 200:400]))
#  	file.close()

thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = np.ones((3,3),np.uint8)
thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
num_components, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh)
a = 0
#for centroid in centroids:
#    a+=1
#    image = cv2.putText(image, str(a), (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
#print('num', num_components-1)
#print('shape:', image.shape)
cv2.imshow('thresh', thresh)

distance_map = ndimage.distance_transform_edt(thresh)

local_max = peak_local_max(distance_map, indices=False, min_distance=1)

# Perform connected component analysis then apply Watershed
markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
labels = watershed(-distance_map, markers, mask=thresh)
plt.imshow(labels)
# with  open(r"C:\Users\Victus\Desktop\file1.txt", "w") as file:
#  	file.write(str(labels))
#  	file.close()
#print(labels.shape)

#print(labels)
a =1
for label in np.unique(labels):                                                                         
    if label == 0:
        continue

    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    # Find contours and determine contour area
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for i in cnts:
        a+=1
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            b, g, r = draw_function(image, cx, cy)
            cv2.drawContours(image, [i], -1, (0, 0, 255), 1)
            text = 'color of colony ' + str(a)+ ': ' + get_color_name(r, g, b) 
            print(text)
            cv2.putText(image, str(a), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        area = cv2.contourArea(i)
        ##print('area of colony',str(a),':', area)
    #c = max(cnts, key=cv2.contourArea)
    #cv2.drawContours(image, [c], -1, (255, 0,0 ), 1)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()
##print('Total colonies:', a)
ax[0].imshow(-distance_map, cmap=plt.cm.gray)
ax[0].set_title('Distances')
ax[1].imshow(labels)
ax[1].set_title('labels')
plt.show()

cv2.imshow('Final', image)
cv2.waitKey(0)
cv2.destroyAllWindows() 