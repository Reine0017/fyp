import sys
import imutils
import cv2
from PIL import Image
import numpy as np

for i in range(27):
    i = str(i)
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
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        resized = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    else:
        resized = img

    image = resized
    grabImg = resized
    mask = np.zeros(resized.shape[:2], np.uint8)

    print(type(image))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(type(gray))

    edged = cv2.Canny(gray, 100, 250)
    cv2.imshow('Edged', edged)
    print(type(edged))

    blur = cv2.bilateralFilter(edged, 9, 90, 90)
    cv2.imshow("blurred", blur)
    print(type(blur))

    kernel = np.ones((5, 5), np.uint8)

    # blur = cv2.erode(blur, kernel, iterations=5)
    # cv2.imshow('thresh1', blur)

    # blur = cv2.dilate(blur, kernel, iterations=5)
    # cv2.imshow('thresh2', blur)

    blur = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)

    im3, cnts1, hierarchy1 = cv2.findContours(blur.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    largestArea0 = 0
    x0 = 0
    y0 = 0
    w0 = 0
    h0 = 0

    for conts in cnts1:
        [x, y, w, h] = cv2.boundingRect(conts)
        rectArea = w * h

        if (rectArea > largestArea0):
            x0 = x
            y0 = y
            w0 = w
            h0 = h
            largestArea0 = w0 * h0

    cv2.rectangle(resized, (x0, y0), (x0 + w0, y0 + h0), (255, 0, 255), 3)
    cv2.imshow("image0", resized)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (x0, y0, w0, h0)

    # Idea: Go into the grabcut source code, change the function such that it iteratively dilated then cut?
    cv2.grabCut(grabImg, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    # Idea: Randomly assign background and foreground pixels?

    cv2.imshow("grabImg", img)

    c = cv2.waitKey(0)
    if (c == 27):
        break
