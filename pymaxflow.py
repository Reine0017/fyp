import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay

from skimage import io
import argparse

#Calculate the SLIC superpixels, their histograms and neighbors
#def superpixels_histograms_neighbors(img):
    #SLIC
#    segments=slic(img, n_segments=500, compactness=20)
#    segments_ids = np.unique(segments)
    
    #centers
    
#img=astronaut()
#segments=slic(img, n_segments=100, compactness=10)

ap=argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="/home/fangran/fyp/resource/images/guppyfish.jpeg")
args=vars(ap.parse_args())

image=img_as_float(io.imread(args["image"]))

for numSegments in (100,200,300):
    segments=slic(image, n_segments=numSegments, sigma=5)

    fig=plt.figure("Superpixels -- %d segments" %(numSegments))
    ax=fig.add_subplot(1,1,1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")

plt.show()
