# import the necessary packages
import numpy as np
import csv

class Calculator:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath

	def chi2_distance(self, histA, histB, eps = 1e-10):
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
		return d

	#limit is the maximum number of results to return.
	def calculate(self, queryFeatures, limit = 10):
		# initialize our dictionary of results
		results = {}

		# open the index file for reading
		with open(self.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)

			# loop over the rows in the index
			for row in reader:
				# parse out the image ID and features, then compute the
				# chi-squared distance between the features in our index
				# and our query features
				features = [float(x) for x in row[1:]]
				d = self.chi2_distance(features, queryFeatures)

				results[row[0]] = d

			f.close()

		results = sorted([(v, k) for (k, v) in results.items()])

		return results[:limit]