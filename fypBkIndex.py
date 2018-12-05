from fypColorDesc import ColorDescriptor
from fypResize0210 import *
import argparse
import glob
import cv2

cvBKFilepath='/home/fangran/fyp/cocoapi-master/backgrounds' + '/*.jpg'
CSVFilepath='/home/fangran/fyp/cocoapi-master/index.csv'

def BKIndex():
	# initialize the color descriptor
	cd = ColorDescriptor((8, 12, 3))

	output = open(CSVFilepath,"w")

	# use glob to grab the image paths and loop over them
	for imagePath in glob.glob(cvBKFilepath):
		# extract the image ID (i.e. the unique filename) from the image
		# path and load the image itself
		imageID = imagePath[imagePath.rfind("/") + 1:]
		image = cv2.imread(imagePath)

		#image=image_resize(image,height=600)
	 
		# describe the image
		features = cd.describe(image)
	 
		# write the features to file
		features = [str(f) for f in features]
		output.write("%s,%s\n" % (imageID, ",".join(features)))
	 
	# close the index file
	output.close()

	print("done")

	return

if __name__ == "__main__":
	BKIndex()