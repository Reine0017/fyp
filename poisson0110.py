#!/usr/bin/env python

import sys
import imutils
import cv2
from PIL import Image
import glob
import numpy as np
#import colour
import os
from getPoints2709 import *
from scipy.spatial import distance
import math

myCats=["bird","boat","car","kite","person","elephant"]

srcResizedH=300
bkResizedH=800

#have this as a separate script since many scripts use it
#resize function
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

cvBKFilepath='/home/fangran/fyp/cocoapi-master/backgrounds' + '/*.jpg'
oriImgFilepath = '/home/fangran/fyp/cocoapi-master/original(new1)/bird' + '/*.jpg'
maskImgFilepath = '/home/fangran/fyp/cocoapi-master/masks(new1)/bird' + '/*.jpg'
filesBK = glob.glob(cvBKFilepath)
filesOri = glob.glob(oriImgFilepath)
filesMask = glob.glob(maskImgFilepath)

for f1 in filesBK:
	img = cv2.imread(f1)

	img = image_resize(img, height=bkResizedH)

	for (f2,f3) in zip(filesOri, filesMask):
		oriImgName = os.path.basename(os.path.normpath(f2))
		maskImgName = os.path.basename(os.path.normpath(f3))
		if(oriImgName != maskImgName):
			myList.append("no")
		oriImage = cv2.imread(f2)
		maskImage0 = cv2.imread(f3)
		#0 parameter reads image as grayscale
		maskImage = cv2.imread(f3, 0)

		print("maskImage shape")
		print(maskImage.shape)

		#Morphological Operations on Masks
		kernel = np.ones((5,5),np.uint8)

		dilatedMask = cv2.dilate(maskImage,kernel,iterations = 3)

		dilatedMask1 = cv2.dilate(maskImage0,kernel,iterations=3)

		#erode my tight mask to make it tighter
		erodedMask = cv2.erode(maskImage,kernel,iterations = 3)

		differenceM = cv2.bitwise_xor(erodedMask.copy(), dilatedMask.copy())

		ret,binaryThreshMask = cv2.threshold(differenceM,100,255,cv2.THRESH_BINARY)

		s = cv2.imshow("differenceMTresh",binaryThreshMask)

		diffMaskNewArray = np.transpose(np.nonzero(binaryThreshMask))

		minDistArrDil=np.zeros(shape=(len(diffMaskNewArray),1)).astype(int)
		minDistArrEro=np.zeros(shape=(len(diffMaskNewArray),1)).astype(int)

		b = getPoints(dilatedMask)
		c = getPoints(erodedMask)

		row=dilatedMask.shape[0]
		col=dilatedMask.shape[1]
		black=np.zeros((row,col,1), dtype="uint8")
		black1=black.copy()

		ohnoCount = 0
		diffMaskNewArrayCount=0

		#seamless clone
		center = (1000,400)

		clonedImg = cv2.seamlessClone(oriImage, img, dilatedMask1, center, cv2.NORMAL_CLONE)
		
		#make copy of poisson blended image
		clonedImgCopy = clonedImg.copy()

		wrongCount=0
		for i in diffMaskNewArray:
			#get min dist bet point and set of points
			#for dil mask
			a = [i]
			#print(a[0][0])

			#print(a[0)
			x=a[0][0]
			y=a[0][1]

			black1[x,y] = [255]

			diffMaskNewArrayCount = diffMaskNewArrayCount+1

			#shortestDistDil is Du
			shortestDistDil = np.amin(distance.cdist(a, b, 'euclidean'))
			#shortestDistDil = round(shortestDistDil,3)
			minDistArrDil[i] = shortestDistDil

			#shortestDistEro is Dobj
			shortestDistEro = np.amin(distance.cdist(a, c, 'euclidean'))
			#shortestDistEro = round(shortestDistEro,3)
			minDistArrEro[i] = shortestDistEro

			#fix divide by zero error
			if (shortestDistEro + shortestDistDil) != 0:
				#distRatio r
				distRatio = shortestDistDil/(shortestDistDil+shortestDistEro)

				#smoothing function
				newDistRatio = (np.sin((np.pi*distRatio) - (np.pi/2)) + 1)/2

				if newDistRatio>1 or (1-newDistRatio)>1:
					wrongCount = wrongCount + 1

				#get b g r value of original Image at this pixel location
				#print("clonedImg[x,y]")
				#print(clonedImg[x,y])
				clonedPix = clonedImg[x,y]

				#returns b value at this pixel
				#print(clonedPix[0])

				#get b g r value of poisson blended Image at this pixel location
				oriPix = oriImage[x,y]

				#calculate new pixel colour of this pixel

				#new b value
				newBPix = clonedPix[0] * (1-newDistRatio) + oriPix[0] * newDistRatio

				#new g value
				newGPix = clonedPix[1] * (1-newDistRatio) + oriPix[1] * newDistRatio

				#new r value
				newRPix = clonedPix[2] * (1-newDistRatio) + oriPix[2] * newDistRatio

				#Assign this pixel a the new pixel colour value for this copy
				clonedImgCopy[x,y] = [newBPix,newGPix,newRPix]

			#elif (shortestDistEro+shortestDistDil)==0:
				#print("OH NOOOOOOOOOOOOOOOOOO")
				#ohnoCount = ohnoCount + 1
				#x_val=a[0][0]
				#y_val=a[0][1]
				#black[x_val, y_val] = [255]

		print(diffMaskNewArrayCount)
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		print(wrongCount)

		#cv2.imshow("black",black)
		#cv2.imshow("1",erodedMask)
		#cv2.imshow("2",dilatedMask)
		#cv2.imshow("3",differenceM)
		#cv2.imshow("black1",black1)

		cv2.imshow("clonedImg", clonedImg)
		cv2.imshow("clonedImgCopy", clonedImgCopy)

		s = cv2.waitKey(0)

		if s == 27:
			sys.exit()