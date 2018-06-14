#!/usr/bin/env python

import os, argparse
from skimage import segmentation
from skimage.future import graph
import cv2, numpy
import tempfile
import random
import code

def color_segments(segments):
    num_segments = segments.max()+1
    colors = numpy.array([(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for i in range(num_segments)])
    return colors[segments]

def centroids(segments):
    assert len(segments.shape) == 2
    num_segments = segments.max()+1
    row_sums = [0 for i in range(num_segments)]
    col_sums = [0 for i in range(num_segments)]
    pixel_counts = [0 for i in range(num_segments)]
    for ((row,col),value) in numpy.ndenumerate(segments):
        row_sums[value] += row
        col_sums[value] += col
        pixel_counts[value] += 1
    result = numpy.zeros(segments.shape,dtype=segments.dtype)
    for i in range(num_segments):
        row = row_sums[i]//pixel_counts[i]
        col = col_sums[i]//pixel_counts[i]
        if result[row,col] == 0:
            result[row,col] = i
        else:
            "(%d,%d) is centroid of multiple segments, ignoring"%(row,col)
    return result

def convert_specks_to_boundaries(segments,min_size=12):
    # Any small segments left from the watershed step get converted to boundary pixels
    # and will later be erased by erase_boundaries
    labels, counts = numpy.unique(segments,return_counts=True)
    small_segs = []
    for i in range(len(labels)):
        if counts[i] < min_size:
            small_segs.append(labels[i])
    small_seg_mask = numpy.in1d(segments.reshape(-1),small_segs).reshape(segments.shape)
    return numpy.where(small_seg_mask,-1,segments)

def erase_boundaries(ws_segments):
    #cv2.watershed leaves -1 values around the edges and between segments
    #Also small segments have been converted to -1
    #replace these with neighboring labels semi-arbitrarily
    result = ws_segments.copy()

    iter_count = 0
    while (result.min() < 0):
        # Shift one pixel in each direction on each pass
        result[result==-1] = numpy.pad(result,((0,0),(0,1)),'edge')[:,1:][result==-1]
        result[result==-1] = numpy.pad(result,((0,1),(0,0)),'edge')[1:][result==-1]
        result[result==-1] = numpy.pad(result,((0,0),(1,0)),'edge')[:,:-1][result==-1]
        result[result==-1] = numpy.pad(result,((1,0),(0,0)),'edge')[:-1][result==-1]
        iter_count += 1
        assert iter_count <= 10, "Too many iterations" # Just in case
    return result

def show_img(img_out):
    tmp = tempfile.NamedTemporaryFile(suffix='.tif',delete=False)
    tmp.close()
    cv2.imwrite(tmp.name,img_out)
    os.system('open '+tmp.name)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input-file')
    
    args = parser.parse_args()
    img = cv2.imread(getattr(args,'input-file'))[:,:,::-1]
    seg_centroids = centroids(segmentation.slic(img,n_segments=5000))
    segments = cv2.watershed(img,seg_centroids.astype(numpy.int32))
    segments = erase_boundaries(convert_specks_to_boundaries(segments))
    
    show_img(color_segments(segments))
    
main()
