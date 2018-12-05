#!/usr/bin/env python

import sys
import imutils
import cv2
from PIL import Image
import numpy as np
import glob
import keyboard
from tkinter import *
from fypResize0210 import *
from fypBkSearch import *

#Initialize
#cvBKFilepath0='/home/fangran/fyp/cocoapi-master/backgrounds/'
#cvBKFilepath='/home/fangran/fyp/cocoapi-master/backgrounds' + '/*.jpg'
#oriImgFilepath = '/home/fangran/fyp/cocoapi-master/original/bird/000000100489.jpg'
#maskImgFilepath = '/home/fangran/fyp/cocoapi-master/masks/bird' + '/*.jpg

drawing = False # true if mouse is pressed
ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
        	cv2.circle(BKimg,(x,y),3,(0,0,0),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(BKimg,(x,y),3,(0,0,0),-1)


def drawingFunction(DrawOnImage,saveFilePath):
	#drawing = False # true if mouse is pressed
	#ix,iy = -1,-1
	imageCopy=DrawOnImage.copy()
	cv2.namedWindow('press s to save')
	cv2.setMouseCallback('press s to save',draw_circle)

	while(1):
		cv2.imshow('press s to save',DrawOnImage)
		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break

		#if user presses 's'
		elif k==ord('s'):
			cv2.imwrite(saveFilePath,DrawOnImage)
			break

	cv2.destroyWindow('press s to save')
	return DrawOnImage

#image = image_resize(BKimg,height=BKImgHeight)
#cv2.imshow("This is the BK Image selected", BKimg)

if __name__ == "__main__":
	oriImgFilepath = '/home/fangran/fyp/cocoapi-master/original/bird/000000100489.jpg'
	#maskImgFilepath = '/home/fangran/fyp/cocoapi-master/masks/bird' + '/*.jpg

	BKimg = cv2.imread(oriImgFilepath, 1)

	clone=BKimg.copy()

	#cv2.namedWindow('draw image')
	#cv2.setMouseCallback('draw image',draw_circle)
	
	imgReturned = drawingFunction(BKimg,'/home/fangran/fyp/cocoapi-master/sketches/000000100489.jpg')
	
	while(1):
		cv2.imshow("Save this image?", imgReturned)
		s=cv2.waitKey(0)

		if s==ord('y'):
			print("IMAGE SAVED")
			break

		if s==ord('n'):
			cv2.destroyWindow('Save this image?')
			BKimg=clone.copy()
			imgReturned=drawingFunction(BKimg,'/home/fangran/fyp/cocoapi-master/sketches/000000100489.jpg')
			#cv2.imshow("img ret1", imgReturned)
			continue

		if s == 27:
			cv2.destroyAllWindows()
			break