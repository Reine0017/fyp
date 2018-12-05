#!/usr/bin/env python

import numpy as np
import cv2
import imutils
import glob
import keyboard
import sys
from fypDraw0310 import *
from getPoints2709 import *

def Draw2MaskPoints(image):
	(B, G, R) = cv2.split(image)
	G[G < 255] = 0

	myPoints=getPoints(G,PIXNO=100,stepflag=1)

	return myPoints

if __name__ == "__main__":
	oriImgFilepath = '/home/fangran/fyp/cocoapi-master/original/bird/000000100489.jpg'
	#maskImgFilepath = '/home/fangran/fyp/cocoapi-master/masks/bird' + '/*.jpg
	fileSaveTo = '/home/fangran/fyp/cocoapi-master/sketches/bk41.png'
	drawnImg = cv2.imread(fileSaveTo, 1)

	#for testing
	'''
	backtorgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
	
	for i in range(len(myPoints)):
		#print(selected_pixels[i][0])
		y_val=myPoints[i][0]
		x_val=myPoints[i][1]
		#finalPoints[i] = [x_val, y_val]
		#finalPoints.append([x_val,y_val])
		backtorgb[y_val, x_val]=[0,255,255]
	'''
	

	print(len(myPoints))
	print(myPoints[0])
	print(myPoints[0][0])

	print(mask.shape)
	cv2.imshow("",mask)

	#testing
	#cv2.imshow("1",backtorgb)

	t=cv2.waitKey(0)

	if t==27:
		sys.exit()