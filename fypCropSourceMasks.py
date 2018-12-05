import cv2
import numpy as np
import glob
import csv

#masksFilepath='/home/fangran/fyp/cocoapi-master/backgrounds' + '/*.jpg'

myCats=["bird","boat","car","kite","person","elephant"]

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

if __name__ == '__main__':
	
	'''
	image=cv2.imread('/home/fangran/fyp/resource/masks/binMaskT7.jpg',0)
	points=getPoints(image,300,1)
	#shapeContext(points)

	print(shapeContext(points))
	'''
	for i in myCats:
		MaskImagePath = '/home/fangran/fyp/cocoapi-master/masks (copy)/' + str(i) + '/*.jpg'
		OriImagePath = '/home/fangran/fyp/cocoapi-master/original (copy)/' + str(i) + '/*.jpg'
		newMaskImagePath='/home/fangran/fyp/cocoapi-master/masks(new1)/' +str(i)
		newOriImagePath = '/home/fangran/fyp/cocoapi-master/original(new1)/' + str(i)

		print(newMaskImagePath)

		#Python: cv2.imwrite(filename, img[, params]) → retval
		#global x1,x2,y1,y2

		for imagePath in glob.glob(MaskImagePath):
			#print(imagePath)
			imageID = imagePath[imagePath.rfind("/") + 1:]
			#print(imageID)

			grayImage = cv2.imread(imagePath, 0)

			height, width = grayImage.shape[:2]

			#cv2.imshow("gray",grayImage)

			(thresh, im_bw) = cv2.threshold(grayImage, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

			cv2.imshow("binary",im_bw)

			#dilate im_bw image
			kernel = np.ones((8,8),np.uint8)
			dilation = cv2.dilate(im_bw,kernel,iterations = 5)

			cv2.imshow("dilate",dilation)

			where = np.array(np.where(dilation))

			x1, y1 = np.amin(where, axis=1)
			x2, y2 = np.amax(where, axis=1)

			print(x1,y1)

			print(x2,y2)
			sub_image = grayImage[x1:x2, y1:y2]

			cv2.imshow("sub_image",sub_image)

			print(sub_image.shape)

			cv2.imwrite(newMaskImagePath + "/" + imageID, sub_image)

			copiedFlag=False

			for imagePath1 in glob.glob(OriImagePath):
				#print(imagePath1)
				imageID1 = imagePath1[imagePath1.rfind("/") + 1:]
				print(imageID1)

				if(copiedFlag==False):
					if (imageID==imageID1):
						myImage = cv2.imread(imagePath1)
						myImage = image_resize(myImage, height=height)
						cv2.imshow("",myImage)
						sub_image1 = myImage[x1:x2, y1:y2]

						cv2.imshow("sub_image1",sub_image1)

						print(sub_image1.shape)
						print("%%%%%%%%%%%%%%%%%%%%%%%")

						cv2.imwrite(newOriImagePath + "/" + imageID1, sub_image1)

						copiedFlag=True
					else:
						continue
				else:
					break


			s = cv2.waitKey(1)

			if s == 27:
				cv2.destroyAllWindows()
				break