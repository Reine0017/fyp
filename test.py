import sys
import imutils
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

#Comment Comment

for i in range (91):
	i = str(i)
	stage1=sys.argv[2] + "image" + i + ".jpg"
	file = sys.argv[1] + i + ".jpg"
	print(file)

	img = cv2.imread(file, 1)
    
	height, width = img.shape[:2]
	max_height = 800
	max_width = 800

	# only shrink if img is bigger than required
	if max_height < height or max_width < width:
    	# get scaling factor
		scaling_factor = max_height / float(height)
		if max_width/float(width) < scaling_factor:
			scaling_factor = max_width / float(width)
		# resize image
		resized = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
	else:
		resized = img

	image=resized
	grabcutImg=resized

	height1,width1 = resized.shape[:2]



	#clustering to extract foreground image?



	#to draw rough bounding box
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imshow("gray", gray)

	edged = cv2.Canny(gray, 80, 250)
	cv2.imshow('Edged', edged)

	blur=cv2.bilateralFilter(edged,5,75,75)
	cv2.imshow("blurred", blur)

	thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15,8)
	cv2.imshow('thresh',  thresh)

	kernel = np.ones((5,5), np.uint8)

	#thresh =cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	#thresh =cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,None)
	#thresh = cv2.erode(thresh, kernel, iterations=10)
	#cv2.imshow('thresh1', thresh)

	thresh = cv2.erode(thresh, None, iterations=15)
	cv2.imshow('thresh1', thresh)

	thresh = cv2.dilate(thresh, None, iterations=15)
	cv2.imshow('thresh2', thresh)
	
	thresh=cv2.bitwise_not(thresh)
	cv2.imshow('threshbit', thresh)

	im2,cnts,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	largestArea=0
	x1=0
	y1=0
	w1=0
	h1=0
	
	for contour in cnts:
		print("hi")
		[x,y,w,h]=cv2.boundingRect(contour)
		print("x",x)
		print("y",y)
		print("w",w)
		print("h",h)

		rectArea = w*h

		if (rectArea > largestArea):
			x1=x
			y1=y
			w1=w
			h1=h
			largestArea=w*h
		
		if (largestArea == (height1 * width1)):
			x1 = x1+10
			y1 = y1+10
			w1 = w1-5
			h1 = h1-5

	cv2.rectangle(image,(x1,y1), (x1+w1, y1+h1), (255,0,255), 5)
	#cv2.imshow("image",image)
	cv2.imshow("image",image)

	cv2.imwrite(stage1, image)
	# show the output image
	#cv2.imshow("Image", image2)

	mask=np.zeros(grabcutImg.shape[:2], np.uint8)

	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)

	rect = (x1,y1,w1,h1)

	cv2.grabCut(grabcutImg,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	grabcutImg = grabcutImg*mask2[:,:,np.newaxis]

	#plt.imshow(grabcutImg)
	#plt.colorbar()
	#plt.show()

	cv2.imshow("grabcutImg", grabcutImg)

	c = cv2.waitKey(0)
	if (c==27):
		break