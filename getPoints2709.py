import cv2
import numpy as np
from lapjv import lapjv
import sys
import glob
import os

#number of pixels to be chosen
#PIXNO=200
#allPixLen=[]
#allFinalPoints=[]
#allFinalPointsLen=[]

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

def getPoints(maskImg,PIXNO=None,stepflag=0):

	#edges=cv2.Canny(maskImg,100,20)
	#cv2.imshow("edges", edges)

	im2, cnts, hierarchy = cv2.findContours(maskImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	#cv2.imshow("im2", im2)

	#row=im2.shape[0]
	#col=im2.shape[1]

	#Reason why pixels cannot be drawn with colour may be because the '3' in the line below wasnt specified
	#black=np.zeros((row,col,3), dtype="uint8")

	#rgb is brg instead
	#contoured=cv2.drawContours(black, cnts, -1, (255,0,0), 1)
	#contouredC=contoured.copy()

	#cv2.imshow("contouredC", contouredC)

	#newList = np.vstack(cnts)
	newList = np.concatenate( cnts, axis=0 )
	#print(contoured.shape)

	#replaced with the concatenate function above
	#list of ALL points of ALL contours
	#all_pixels=[]

	'''
	for i in range(0, len(cnts)):
		for j in range(0,len(cnts[i])):
			#somehow the values of x and y got flipped??
			#print(cnts[i][j][0])
			#np.append(selected_pixels,cnts[i][j])
			all_pixels.append(cnts[i][j])

	print(len(all_pixels))
	'''
	#print("LENGTH OF newList")
	#print(len(newList))

	#print(newList[0])

	if (stepflag==1):
		newPoints=[]
		step=round(len(newList) / PIXNO)
		for i in range(0,len(newList),step):
			#print(all_pixels[i])
			newPoints.append(newList[i])
			#print("Len(newPoints)")
			#print(len(newPoints))
	else:
		newPoints = newList

	#print("!!!!!!!!!!!!")
	#print(len(newPoints))
	#print(newPoints[0])

	newPointsArray = np.array(newPoints)

	#print("!!!!!!!!!!!!")
	#print(len(newPointsArray))
	#print(newPointsArray[0])
	#To fix: x and y values swapped
	finalPoints=np.zeros(shape=(len(newPointsArray),2)).astype(int)
		
	for i in range(len(newPointsArray)):
		#print(selected_pixels[i][0])
		y_val=newPointsArray[i][0][0]
		x_val=newPointsArray[i][0][1]
		finalPoints[i] = [x_val, y_val]
		#finalPoints.append([x_val,y_val])
		#TESTING
		#contouredC[x_val, y_val]=[255,255,255]
		#print(finalPoints)
		#s = cv2.imshow("smallConts", contouredC)

	#s = cv2.waitKey(0)
	#if s == 27:
	#	sys.exit()

	return finalPoints

if __name__ == '__main__':
    # test1.py executed as script
    # do something
	maskImgFilepath = '/home/fangran/fyp/cocoapi-master/masks/bird' + '/*.jpg'
	filesMask = glob.glob(maskImgFilepath)

	for f1 in filesMask:
		print(f1)
		grayImage = cv2.imread(f1, 0)
		grayImage = image_resize(grayImage, height=500)

		#this 255 can be 1 instead
		grayImage[grayImage!=0] = 255


		myPoints = getPoints(grayImage)

		print(len(myPoints))

		'''
		print("**************************")
		print(len(myPoints))
		print(myPoints[0])
		print(myPoints[0][0])
		print(myPoints)
		print(type(myPoints))
		'''