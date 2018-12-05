# import the necessary packages
import numpy as np
import cv2
import imutils

class ColorDescriptor:
	def __init__(self, bins):
		# store the number of bins for the 3D histogram
		self.bins = bins

	def histogram(self,image,mask):
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,[0, 180, 0, 256, 0, 256])
		hist=cv2.normalize(hist,hist).flatten()

		return hist

	def describe(self,image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []

		# grab the dimensions and compute the center of the image
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))

		# divide the image into four rectangles/segments (top-left,
		# top-right, bottom-right, bottom-left)
		segments = [(0,0,cX,cY), (cX,0,w,cY),(0,cY,cX,h),(cX,cY,w,h)]

		# construct an elliptical mask representing the center of the image
		(axesX, axesY) = (int(w * 0.70) // 2, int(h * 0.70) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

		# loop over the segments
		for (startX, startY, endX, endY) in segments:
			# construct a mask for each corner of the image, subtracting the elliptical center from it
			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)

			hist = self.histogram(image, cornerMask)
			features.extend(hist)

		hist=self.histogram(image,ellipMask)
		features.extend(hist)

		return features