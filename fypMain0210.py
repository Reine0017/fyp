#!/usr/bin/env python

import pdb
import os
import sys
import imutils
import cv2
#from PIL import Image
import numpy as np
import glob
import keyboard
#from tkinter import *
from fypResize0210 import *
from fypBkSearch import *
from fypDraw2Mask import *
import fypHungarian
#from fypHungarian import *

#Initialize
cvBKFilepath0='/home/fangran/fyp/cocoapi-master/backgrounds/'
cvBKFilepath='/home/fangran/fyp/cocoapi-master/backgrounds' + '/*.jpg'
#oriImgFilepath = '/home/fangran/fyp/cocoapi-master/original/bird' + '/*.jpg'
#maskImgFilepath = '/home/fangran/fyp/cocoapi-master/masks/bird' + '/*.jpg'
sketchesFilepath = '/home/fangran/fyp/cocoapi-master/sketches/'

BKImgHeight = 700

##################
#
#Helper Methods
#
##################
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
        	cv2.circle(BKimg,(x,y),3,(0,255,0),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(BKimg,(x,y),3,(0,255,0),-1)


def drawingFunction(DrawOnImage,saveFilePath):
	#drawing = False # true if mouse is pressed
	#ix,iy = -1,-1
	#imageCopy=DrawOnImage.copy()
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

def max_rgb_filter(image):
	# split the image into its BGR components
	(B, G, R) = cv2.split(image)
 
	#in the newly returned image, there are no pixels with ONLY green value(=255)
	B[B > 225] = 225
	G[G > 225] = 225
	R[R > 225] = 225
 
	# merge the channels back together and return the image
	return cv2.merge([B, G, R])

###########################################################################################
#
#Section 1
#
########################################################################################1##

backPic = 0
backImgSelected = None
#BKImgs = [cv2.imread(file) for file in glob.glob(cvBKFilepath)]
#print(len(BKImgs))
#print(BKImgs[0])

filesBK = glob.glob(cvBKFilepath)
print(len(filesBK))
print(filesBK[0])

backPicsTotal = len(filesBK)

BKImgQueried = ""
BKImg = ""

while(backPic<=backPicsTotal):
	if keyboard.is_pressed('right'):
		print("right is pressed, showing next background image")
		backPic = backPic + 1

	elif keyboard.is_pressed('left'):
		print("left is pressed, showing previous background image")
		if (backPic!=0):
			backPic = backPic - 1
		else:
			print("this is the first image!")

	elif keyboard.is_pressed('enter'):
		print("This image is selected")
		print(filesBK[backPic])
		BKImgQueried = filesBK[backPic]
		cv2.destroyWindow("background image")
		break

	print(backPic)
	img = cv2.imread(filesBK[backPic], 1)

	image = image_resize(img,height=BKImgHeight)
	cv2.imshow("background image", image)

	s = cv2.waitKey(0)
	if s == 27:
		break

#print("THIS WAS THE IMAGE YOU SELECTED")

#img = cv2.imread(BKImgQueried, 1)

#image = image_resize(img,height=BKImgHeight)
#cv2.imshow("background image query", image)

print("PLEASE WAIT WHILE SIMILAR IMAGES ARE RETRIEVED")
testResults=selectTopImage(image)

backPicsTotal1=len(testResults)
backPic1=0

while(backPic1<=backPicsTotal1):
	if keyboard.is_pressed('right'):
		print("right is pressed, showing next background image")
		backPic1 = backPic1 + 1

	elif keyboard.is_pressed('left'):
		print("left is pressed, showing previous background image")
		if (backPic1!=0):
			backPic1 = backPic1 - 1
		else:
			print("this is the first image!")
	elif keyboard.is_pressed('enter'):
		print("This image is selected (FINAL)")
		print(currentImg)
		BKImg = currentImg
		cv2.destroyWindow("choose background image FINAL")
		break

	currentImg = cvBKFilepath0+testResults[backPic1][1]
	img = cv2.imread(currentImg, 1)

	image = image_resize(img,height=BKImgHeight)
	cv2.imshow("choose background image FINAL", image)

	s = cv2.waitKey(0)
	if s == 27:
		break

print(BKImg)

###########################################################################################
#
#Section 2
#
###########################################################################################

toDraw=input("What do you want to draw? ")

print(toDraw)

BKimg = cv2.imread(BKImg, 1)

BKimg = image_resize(BKimg,height=BKImgHeight)
#cv2.imshow("This is the BK Image selected", BKimg)

print("!!!!!!!!!!!!!!!!!!!")
print(os.path.basename(os.path.normpath(BKImg)))
fileLastP = os.path.basename(os.path.normpath(BKImg))
fileSaveTo = sketchesFilepath + fileLastP
print(fileSaveTo)

#imgDrawnOn=drawingFunction(BKimg,)

s = cv2.waitKey(0)

BKimg = max_rgb_filter(BKimg)

#if s==ord('d'):
clone=BKimg.copy()
imgDrawnOn=drawingFunction(BKimg,fileSaveTo)

while(1):
	cv2.imshow("Save this image?", imgDrawnOn)
	s=cv2.waitKey(0)

	if s==ord('y'):
		print("IMAGE SAVED")
		break

	if s==ord('n'):
		cv2.destroyWindow('Save this image?')
		BKimg=clone.copy()
		imgDrawnOn=drawingFunction(BKimg,fileSaveTo)
		#cv2.imshow("img ret1", imgReturned)
		continue

	if s == 27:
		cv2.destroyAllWindows()
		break


#if s == 27:
#	sys.exit()

#imgDrawnOn is saved
#cv2.imshow("",imgDrawnOn)

#pass imgDrawnOn and path to draw2MaskPoints function
drawingPoints = Draw2MaskPoints(imgDrawnOn)

#Testing
print("LENGTH OF DRAWING POINTS & DRAWING POINTS")
print(len(drawingPoints))
print(type(drawingPoints))
print(drawingPoints)

print(type(toDraw))
print(toDraw)

cv2.destroyAllWindows()

CSVFilepath='/home/fangran/fyp/cocoapi-master/'+toDraw+ 'Index3.csv'

bridgeFilepath='/home/fangran/fyp/cocoapi-master/bridge.csv'

np.savetxt(bridgeFilepath,drawingPoints, fmt='%u')

#myFinalArray1 = fypHungarian.getCompArray(CSVFilepath,toDraw,drawingPoints)

#print(myFinalArray1)

#JUST FOR ME TO SEE, ACTUAL APP NOT USED
height,width,channels=BKimg.shape
blank_image = np.zeros((height,width,1), np.uint8)
for i in range(len(drawingPoints)):
	x_val=drawingPoints[i][0]
	y_val=drawingPoints[i][1]
	blank_image[x_val, y_val]=[255]

cv2.imshow("blank_image", blank_image)
where = np.array(np.where(blank_image))
print(type(blank_image))
print(blank_image.shape)
print(where)

cv2.waitKey(0)

#THIS IS TO GET THE SIZE OF SOURCE IMAGE AND POSITION IN BACKGROUND TO BE BLENDED
y1, x1 = np.amin(drawingPoints, axis=0)
y2, x2 = np.amax(drawingPoints, axis=0)

print(x1,y1)

print(x2,y2)

#get size of this rect and mid point
width = x2-x1
height=y2-y1

midPointx = x1 + ((x2-x1)//2)
midPointy = y1 + ((y2-y1)//2)

print(width)
print(height)

print(midPointx)
print(midPointy)