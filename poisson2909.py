#!/usr/bin/env python

import sys
import imutils
import cv2
from PIL import Image
import glob
import numpy as np
import colour
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


#returns 2d array (same shape as image)
def calcMinDist(point,myMask):
	boundaryPoints = getPoints(myMask)
	print(type(point))
	print(point.shape)
	print(type(boundaryPoints))
	print(boundaryPoints.shape)
	return

cvBKFilepath='/home/fangran/fyp/cocoapi-master/backgrounds' + '/*.jpg'
oriImgFilepath = '/home/fangran/fyp/cocoapi-master/original/bird' + '/*.jpg'
maskImgFilepath = '/home/fangran/fyp/cocoapi-master/masks/bird' + '/*.jpg'
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

		#test to see if morph functions will affect original img
		#s = cv2.imshow("myMask0", maskImage0)

		dilatedMask = cv2.dilate(maskImage,kernel,iterations = 3)
		#tested that dilate function doesn't change original image
		#s = cv2.imshow("myMaskD", maskImage0)

		#erode my tight mask to make it tighter
		erodedMask = cv2.erode(maskImage,kernel,iterations = 3)
		#test and see how the eroded mask looks like
		#s = cv2.imshow("erodedM", erodedMask)
		#s = cv2.imshow("myMaskE", maskImage0)

		#Get binary mask of values 0 and 1
		#ret,binaryThreshMask = cv2.threshold(maskImage,127,1,cv2.THRESH_BINARY)
		#ret,binaryThreshMaskDil = cv2.threshold(dilatedMask,1,1,cv2.THRESH_BINARY)
		#ret,binaryThreshMaskEro = cv2.threshold(erodedMask,1,1,cv2.THRESH_BINARY)

		#s = cv2.imshow("binDil",binaryThreshMaskDil)
		#s = cv2.imshow("binEro", binaryThreshMaskEro)

		#print(binaryThreshMask[345])
		#print(binaryThreshMask[5])

		#img_out = np.where(maskImage0,(oriImage*0.5).astype(int),oriImage)
		differenceM = cv2.bitwise_xor(erodedMask.copy(), dilatedMask.copy())


		#for testing
		'''
		row=maskImage.shape[0]
		col=maskImage.shape[1]
		black=np.zeros((row,col,1), dtype="uint8")
		
		print(len(differenceM))
		print(differenceM.shape)


		#diffMaskNewArray saves the indexes of the white pixels

		#print(differenceM[0])
		#print(differenceM[0][0])
		#print(type(differenceM))

		myCount=0
		print(differenceM.shape)
		print(differenceM.shape[0])
		for i in range(differenceM.shape[0]):
			for j in range(differenceM.shape[1]):
				if differenceM[i][j]!=0:
					myCount = myCount + 1
					#print(differenceM[i][j])
		'''
		ret,binaryThreshMask = cv2.threshold(differenceM,100,255,cv2.THRESH_BINARY)

		'''
		myCount1=0
		print(binaryThreshMask.shape)
		print(binaryThreshMask.shape[0])
		for i in range(binaryThreshMask.shape[0]):
			for j in range(binaryThreshMask.shape[1]):
				if binaryThreshMask[i][j]!=0:
					if binaryThreshMask[i][j]!=255:
						myCount1 = myCount1 + 1
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print(myCount1)
		'''

		s = cv2.imshow("differenceMTresh",binaryThreshMask)

		diffMaskNewArray = np.transpose(np.nonzero(binaryThreshMask))

		minDistArrDil=np.zeros(shape=(len(diffMaskNewArray),1)).astype(int)
		minDistArrEro=np.zeros(shape=(len(diffMaskNewArray),1)).astype(int)

		#print(len(diffMaskNewArray))
		#print(minDistArrDil.shape)
		#print(minDistArrDil)

		b = getPoints(dilatedMask)
		c = getPoints(erodedMask)
		#print(b)

		row=dilatedMask.shape[0]
		col=dilatedMask.shape[1]
		black=np.zeros((row,col,1), dtype="uint8")
		black1=black.copy()

		ohnoCount = 0
		diffMaskNewArrayCount=0
		
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

				#get b g r value of original Image at this pixel location

				#get b g r value of poisson blended Image at this pixel location

				#calculate new pixel colour of this pixel

				#make copy of poisson blended image

				#Assign this pixel a the new pixel colour value for this copy

			elif (shortestDistEro+shortestDistDil)==0:
				print("OH NOOOOOOOOOOOOOOOOOO")
				#ohnoCount = ohnoCount + 1
				#x_val=a[0][0]
				#y_val=a[0][1]
				#black[x_val, y_val] = [255]

		print(diffMaskNewArrayCount)

		cv2.imshow("black",black)
		cv2.imshow("1",erodedMask)
		cv2.imshow("2",dilatedMask)
		cv2.imshow("3",differenceM)
		cv2.imshow("black1",black1)

		'''
		#for testing

		print(len(minDistArrDil))
		print(len(minDistArrEro))
		print(type(minDistArrDil[0]))
		print(minDistArrDil[0])
		print(len(diffMaskNewArray))
		

		#print(len(diffMaskNewArray))
		#print(diffMaskNewArray.shape)

		#position of the first pixel
		#print(diffMaskNewArray[0])

		#print(type(diffMaskNewArray))

		#for testing
		#draw on blkImg to check if my pixels are correct.
		for i in range(len(diffMaskNewArray)):
			rowPos = diffMaskNewArray[i][0]
			colPos = diffMaskNewArray[i][1]

			black[rowPos, colPos]=[255]
		'''


		
		#img_out = np.where(differenceM,allMinDistDil(differenceM,dilatedMask),blkImg)

		#s = cv2.imshow("differences",differenceM)
		#s = cv2.imshow("", black)
		#s = cv2.imshow("img_out", img_out)

		s = cv2.waitKey(0)

		if s == 27:
			sys.exit()