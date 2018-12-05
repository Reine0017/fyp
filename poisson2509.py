#!/usr/bin/env python

import sys
import imutils
import cv2
from PIL import Image
import glob
import numpy as np
import colour
import os

myCats=["bird","boat","car","kite","person","elephant"]

srcResizedH=300
bkResizedH=800

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

		kernel = np.ones((5,5),np.uint8)

		dilatedMask = cv2.dilate(maskImage0,kernel,iterations = 3)
		#0 parameter reads image as grayscale
		maskImage = cv2.imread(f3, 0)
		ret,binaryThreshMask = cv2.threshold(maskImage,127,1,cv2.THRESH_BINARY)

		#binaryThreshMask = cv2.dilate(binaryThreshMask,kernel,iterations = 3)

		s =cv2.imshow("thresh_img", maskImage)

		#print(type(binaryThreshMask))
		#print(binaryThreshMask[300])

		where = np.array(np.where(maskImage))

		y1, x1 = np.amin(where, axis=1)
		y2, x2 = np.amax(where, axis=1)

		print(x1, y1)
		print(x2, y2)

		rectWidth = x2 - x1

		rectHeight = y2 - y1
		rectImage = cv2.rectangle(oriImage.copy(),(x1,y1),(x2,y2),(0,255,0),3)

		rectangle = (x1, y1, rectWidth, rectHeight)

		s = cv2.imshow("mask",rectImage)

		#s = cv2.imshow("mask",maskImage)

		#resImage = cv2.bitwise_and(oriImage, dilatedMask)

		#tightMaskImage = cv2.bitwise_and(oriImage, maskImage0)

		#s = cv2.imshow("tightMaskImage", tightMaskImage)

		bgdModel = np.zeros((1,65),np.float64)
		fgdModel = np.zeros((1,65),np.float64)

		#cv2.grabCut(oriImage, binaryThreshMask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
		#cv2.grabCut(resImage, binaryThreshMask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

		#mask2 = np.where((binaryThreshMask==2)|(binaryThreshMask==0),0,1).astype('uint8')

		#output = oriImage*mask2[:,:,np.newaxis]

		center = (1000,400)

		output0 = cv2.seamlessClone(oriImage, img, maskImage0, center, cv2.NORMAL_CLONE)
		output1= cv2.seamlessClone(oriImage, img, dilatedMask, center, cv2.NORMAL_CLONE)

		mask = np.zeros(output1.shape[:2],dtype = np.uint8)

		x0 = 1000 - int(rectWidth/2)
		y0 = 400 - int(rectHeight/2)

		rectangle = (x0-50, y0-50, rectWidth+50, rectHeight+50)

		cv2.grabCut(output1, mask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
		mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
		output = output1*mask2[:,:,np.newaxis]

		#s = cv2.imshow("result",resImage)
		s = cv2.imshow("output", output)
		s = cv2.imshow("maskHi", dilatedMask)
		s = cv2.imshow("output1", output1)
		s = cv2.imshow("output0", output0)
		#s = cv2.imshow("tightMaskImage", tightMaskImage)

		s = cv2.waitKey(0)

		if s == 27:
			sys.exit()