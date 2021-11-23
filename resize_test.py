# Script to resize non-square images, maintaining aspect ratio and centralising

import cv2
import numpy as np

if 1:
	img = cv2.imread('042144-SEB.jpg')
	# get size
	height, width, channels = img.shape
	print (height, width, channels)
	# Create a black image
	x = height if height > width else width
	y = height if height > width else width
	square = np.zeros((x, y, 3), np.uint8)

	square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
	cv2.imshow("original", img)
	cv2.imshow("squared", square)
	cv2.waitKey(0)