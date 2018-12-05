from getPoints2709 import *
from scipy.spatial.distance import cdist, cosine, pdist
import numpy as np
import math
from lapjv import lapjv
import glob
import csv

nbins_r        = 5                   # number of radial bins
nbins_theta    = 12                  # number of angular
r_inner        = 0.125               # inner radius
r_outer        = 2.0                 # outer radius
nbins          = nbins_theta*nbins_r # total number of bins

cvBKFilepath='/home/fangran/fyp/cocoapi-master/backgrounds' + '/*.jpg'

myCats=["bird","boat","car","kite","person","elephant"]

def get_angle(p1,p2):
	# compute the angle between points.
	return math.atan2((p2[1] - p1[1]),(p2[0] - p1[0]))

def angleM(points):
	pointsNum=len(points)
	angleMat = np.zeros((pointsNum, pointsNum))
	for i in range(pointsNum):
		for j in range(pointsNum):
			angleMat[i,j] = get_angle(points[i],points[j])
	return angleMat

def shapeContext(points):
	pointsNum=len(points)
	distMat=cdist(points,points)

	meanDist = distMat.mean()

	distMat_N = distMat/meanDist

	r_bin_edges=np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r)

	# Compute the matrix of labels depending on the location of the points within the radial bins
	r_bin_matrix = np.zeros((pointsNum,pointsNum), dtype=int)

	for r in range(nbins_r):
		r_bin_matrix = r_bin_matrix +  (distMat_N < r_bin_edges[r])

	# boolean indicating points within the region of interest as defined by the radial bins
	r_bool = r_bin_matrix > 0

	# angular matrix
	am = angleM(points)
	#print("am")
	#print(am)
	# Ensure all angles are between 0 and 2Pi
	am_pi = am + 2*math.pi * (am < 0)
	# from angle value to angle bin
	a_bin_matrix = (1 + np.floor(am_pi /(2 * math.pi / nbins_theta))).astype('int')

	BH  = np.zeros(pointsNum*nbins)

	descriptor = np.zeros((pointsNum, nbins))

	for i in range(pointsNum):
		sm = np.zeros((nbins_r, nbins_theta))
		for j in range(pointsNum):
			if (r_bool[i, j]):
				sm[r_bin_matrix[i, j] - 1, a_bin_matrix[i, j] - 1] += 1
				#SCM[i,:,:] = sm
		#print("sm")
		#print(sm)
		BH[i*nbins:i*nbins+nbins] = sm.reshape(nbins)
		descriptor[i]=sm.reshape(nbins)

	#print(descriptor)
	#print(descriptor[0])

	return descriptor



if __name__ == '__main__':
	
	'''
	image=cv2.imread('/home/fangran/fyp/resource/masks/binMaskT7.jpg',0)
	points=getPoints(image,300,1)
	#shapeContext(points)

	print(shapeContext(points))
	'''
	for i in myCats:
		MaskImagePath = '/home/fangran/fyp/cocoapi-master/masks(new1)/' + str(i) + '/*.jpg'

		
		CSVFilepath='/home/fangran/fyp/cocoapi-master/' +str(i)+ 'Index3.csv'
		print(CSVFilepath)

		#1510
		output = open(CSVFilepath,"w")


		for imagePath in glob.glob(MaskImagePath):
			print(imagePath)
			imageID = imagePath[imagePath.rfind("/") + 1:]
			print(imageID)

			grayImage = cv2.imread(imagePath, 0)

			myPoints = getPoints(grayImage,100,1)

			myDescriptor = shapeContext(myPoints)

			print("myDescriptor shape")

			print(myDescriptor.shape[0])

			print(len(myDescriptor))
			print("myDescriptor")
			#print(myDescriptor)

			print(type(myDescriptor))

			print("myPoints")
			print(len(myPoints))

			newDesc = myDescriptor.flatten()

			print(newDesc.shape)

			newDesc = [str(f) for f in newDesc]

			output.write("%s,%s,%s\n" % (imageID,str(myDescriptor.shape[0]),",".join(newDesc)))

		output.close()

		print("done")