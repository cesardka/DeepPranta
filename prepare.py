import os
import numpy as np
import cv2
import cv2 as cv
from matplotlib import pyplot as plt

def detectGreen(src): 
    ## mask of green (36,25,25) ~ (86, 255,255)
    blurred = cv2.bilateralFilter(src,10,75,75)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (22, 25, 25), (80, 255,255))
    mask = cv2.medianBlur(mask, 9)
    filtered = cv.bitwise_and(src,src,mask = mask)
    # plt.subplot(121),plt.imshow(src),plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(filtered),plt.title('Averaging')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    return filtered

def detectEdges(img):
    blurred_float = img.astype(np.float32) / 255.0
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
    edges = edgeDetector.detectEdges(blurred_float) * 255.0
    edges = cv2.bilateralFilter(edges,10,75,75)
    return edges

def filterOutSaltPepperNoise(edgeImg):
    # Get rid of salt & pepper noise.
    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 5)
    while not np.array_equal(lastMedian, median):
        # get those pixels that gets zeroed out
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0

        count = count + 1
        if count > 70:
            break
        lastMedian = median
        median = cv2.medianBlur(edgeImg, 5)

def findSignificantContour(edgeImg):
    contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
        # Find level 1 contours
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)
# From among them, find the contours with large surface area.
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])
    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour

def prepare(path, dest):
    src = cv2.imread(path, 1)
    filtered = detectGreen(src)
    print('save', dest)
    return print(cv2.imwrite(dest, filtered))
    edges = detectEdges(filtered)
    edges_8u = np.asarray(edges, np.uint8)
    filterOutSaltPepperNoise(edges_8u)

    contour = findSignificantContour(edges_8u)
    # Draw the contour on the original image
    contourImg = np.copy(src)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    # cv2.imwrite('contour.jpg', contourImg)

    mask = np.zeros_like(edges_8u)
    cv2.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    # mark inital mask as "probably background"
    # and mapFg as sure foreground
    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD

    # visualize trimap
    trimap_print = np.copy(trimap)
    trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
    trimap_print[trimap_print == cv2.GC_FGD] = 255
    # cv2.imwrite('trimap.png', trimap_print)


    filtered = cv.bitwise_and(src,src,mask = trimap_print)

    # plt.subplot(121),plt.imshow(src),plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(filtered),plt.title('Averaging')
    # plt.xticks([]), plt.yticks([])
    # plt.show()



    # run grabcut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    cv2.grabCut(src, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # create mask again
    mask2 = np.where(
        (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')

    contour2 = findSignificantContour(mask2)
    mask3 = np.zeros_like(mask2)
    cv2.fillPoly(mask3, [contour2], 255)


    # blended alpha cut-out
    mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
    mask4 = cv2.GaussianBlur(mask3, (3, 3), 0)
    alpha = mask4.astype(float) * 1.1  # making blend stronger
    alpha[mask3 > 0] = 255.0
    alpha[alpha > 255] = 255.0

    foreground = np.copy(src).astype(float)
    foreground[mask4 == 0] = 0
    background = np.ones_like(foreground, dtype=float) * 255.0

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha / 255.0
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)
    # Add the masked foreground and background.
    cutout = cv2.add(foreground, background)

    print('save', dest)
    print(cv2.imwrite(dest, cutout))


rootdir = './data/train/'
roomdest = './'
for root, subdirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith('png'):
            label = root.split('/')[-1]
            dirname = os.path.join(roomdest, 'done', 'train', label)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            prepare(os.path.join(root, file), os.path.join(dirname, file))

