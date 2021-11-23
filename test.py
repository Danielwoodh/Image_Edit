import cv2
import numpy as np

img = cv2.imread('042144-SEB.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

cv2.imshow('Sobel XY', sobelxy)
cv2.waitKey(0)

kernel = np.ones((30, 30), np.uint8)
closing = cv2.morphologyEx(img_blur, cv2.MORPH_CLOSE, kernel)

edges = cv2.Canny(image=closing, threshold1=100, threshold2=150)

cv2.imshow('Canny', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()