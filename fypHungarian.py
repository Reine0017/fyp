import cv2
import numpy as np
import math
from lapjv import lapjv
import glob
import csv
#import pandas
#import json
from fypShapeContext import *
from getPoints2709 import *

#from scipy.optimize import linear_sum_assignment

myCats=["bird","boat","car","kite","person","elephant"]

nbins=60

def cost(qAr,dbAr):
	cellCost=0
	for i in range(nbins):
		numerator=(qAr[i]-dbAr[i])**2
		denom=qAr[i]+dbAr[i]
		#should i just +1 to denominator instead?

		#check for divide by zero error
		if (denom):
			cellCost=cellCost+(numerator/denom)

	finalcellCost=round((cellCost*0.5),3)
	return finalcellCost

'''
def hungarian(cost_matrix):
	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	total = cost_matrix[row_ind, col_ind].sum()
	colIndx=col_ind.tolist()
	return colIndx,total
'''

def getCompArray(CSVFilepath,category, qPoints):
	#CSVFilepath='/home/fangran/fyp/cocoapi-master/' +category+ 'Index3.csv'
	
	myPath = CSVFilepath
	finalArrayRet=None
	qShapeC = shapeContext(qPoints)

	with open(myPath,'r') as file:
		reader = csv.reader(file)

		myFinalArray=None
		ImgCount=0

		dtype = [('imgNo','U16'), ('matchCost', float)]
		values=[]

		allTotalCost=[]
		for row in reader:
			#print(row[0])
			#print(row[1])

			flatAr = np.array([float(x) for x in row[2:]])
			#print(type(flatAr))
			dbShapeC = np.reshape(flatAr,(int(row[1]),60))
			#print(dbShapeC.shape)
			#print(dbShapeC[:15])

			#print(qShapeC.shape)
			#print("!!!!!!!!!")

			while(len(qShapeC)!=len(dbShapeC)):
				dbPointsLen = dbShapeC.shape[0]
				qPointsLen = qShapeC.shape[0]
				count=0

				if dbPointsLen > qPointsLen:
					qDBDiff = dbPointsLen - qPointsLen
					smallerSCLen = qShapeC.shape[0]
					#print(qDescriptor.shape[0])
					print("padding qShapeC")
					step = round(smallerSCLen/qDBDiff)
					for i in range(0,qShapeC.shape[0],step):
						qShapeC = np.vstack([qShapeC,qShapeC[i]])
						count = count + 1
						if count==qDBDiff:
							break


				elif qPointsLen > dbPointsLen:
					qDBDiff = qPointsLen - dbPointsLen
					smallerSCLen = dbShapeC.shape[0]
					print("padding dbShapeC")
					step = round(smallerSCLen/qDBDiff)
					for i in range(0,dbShapeC.shape[0],step):
						dbShapeC = np.vstack([dbShapeC,dbShapeC[i]])
						count = count + 1
						if count==qDBDiff:
							break

				else:
					qDBDiff = 0

				print(len(qShapeC))
				print(len(dbShapeC))

			costMatrix=np.zeros(shape=(len(qShapeC),len(qShapeC)))
			for i in range(len(qShapeC)):
				tempRow=[]
				for j in range(len(dbShapeC)):
					cellCost = cost(qShapeC[i],dbShapeC[j])
					#print(cellCost)
					#print(qShapeC[i])
					#print(dbShapeC[j])
					tempRow.append(float(cellCost))
					
				#print(len(tempRow))
				#print(tempRow[:10])
				costMatrix[i] = tempRow
			#print(costMatrix)
			#print(costMatrix.shape)

			#row_ind, col_ind = hungarian(costMatrix)

			row_ind, col_ind, _ = lapjv(costMatrix)
			#print(row_ind)

			row_ind0 = np.arange(len(row_ind))
			totalCost = 0

			#remember dont zip (row_ind, col_ind)
			results = zip(row_ind0,row_ind)

			#dtype = [('imgNo', str), ('matchCost', float)]
			#values=[]

			resCount=0
			for i in results:
				#print(costMatrix.item(i))

				#totalCost of EACH dbImg MUST be calculated.
				#totalCost is cost of matching for this dbImg against qImg
				totalCost += costMatrix.item(i)

			#print("totalCost")
			#print(round(totalCost,3))

			print("Image done")
			print(row[0])
			ImgCount += 1
			print("ImgCount")
			print(ImgCount)

			allTotalCost.append((row[0],round(totalCost,3)))

		#print(allTotalCost)

		#print("!!!!!!!!!!!!!!!")
		#print(row[0])
		#values.append((row[0],totalCost))
		#values.append([row[0],totalCost])
		#print(values)

		#print("VALUES")
		#print(values)
		finalArray = np.array(allTotalCost,dtype=dtype)
		myFinalArray = np.sort(finalArray,order='matchCost')

		#print(myFinalArray)

		finalArrayRet = myFinalArray

	return finalArrayRet


if __name__ == '__main__':
	#queryFile = '/home/fangran/fyp/cocoapi-master/masks(new1)/bird/' + '000000270705.jpg'

	#grayImage = cv2.imread(queryFile, 0)

	#qPoints = getPoints(grayImage, PIXNO=100,stepflag=1)

	#print(qPoints)

	bridgeFilepath='/home/fangran/fyp/cocoapi-master/bridge.csv'

	myPoints = np.loadtxt(bridgeFilepath, dtype='int')

	print(type(myPoints))
	print(len(myPoints))
	print(myPoints)

	CSVFilepath='/home/fangran/fyp/cocoapi-master/' +'bird'+ 'Index3.csv'
	myFinalArray1 = getCompArray(CSVFilepath,'bird',myPoints)

	print(myFinalArray1)

	print(type(myFinalArray1))

	bridgeFilepath1='/home/fangran/fyp/cocoapi-master/bridge1.csv'

	np.save(bridgeFilepath1,myFinalArray1)