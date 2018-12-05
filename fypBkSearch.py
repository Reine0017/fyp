# import the necessary packages
from fypColorDesc import ColorDescriptor
from fypBkCalc import Calculator
import cv2
from fypResize0210 import *
import glob
import keyboard
from tkinter import *

CSVFilepath='/home/fangran/fyp/cocoapi-master/index.csv'
cvBKFilepath='/home/fangran/fyp/cocoapi-master/backgrounds/'
BKImgHeight=700

def selectTopImage(queryImg):
	# initialize the image descriptor
	cd = ColorDescriptor((8, 12, 3))

	# load the query image and describe it
	#query = cv2.imread(args["query"])
	#query=image_resize(query,height=600)
	query = queryImg

	features = cd.describe(query)

	# perform the search
	calculator = Calculator(CSVFilepath)
	results = calculator.calculate(features)

	'''
	# display the query
	cv2.imshow("Query", query)

	selectResultID=""
	# loop over the results
	for (score, resultID) in results:
		# load the result image and display it
		result = cv2.imread(cvBKFilepath + resultID)
		result=image_resize(result,height=BKImgHeight)
		cv2.imshow("Result", result)
		cv2.waitKey(0)
		print(score)
		print(resultID)


	
		if keyboard.is_pressed('enter'):
			print("This image is selected")
			selectResultID=resultID
			print("SELECTED RESULT ID")
			print(selectResultID)
			break
	'''

	return results