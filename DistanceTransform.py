import sys

import cv2
import numpy as np

IMAGE_SRC = sys.argv[1]
THRESHOLD = int(sys.argv[2])
TRANSFORM_TYPE = sys.argv[3]
OUTPUT_FILE = sys.argv[4] if len(sys.argv) == 5 else None
CONNECTIVITY = 4

# read image
img = cv2.imread(IMAGE_SRC)

# convert to grayscale
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert to binary
_, thresh = cv2.threshold(grey, 127, 255, 0)

HEIGHT = thresh.shape[0]
WIDTH = thresh.shape[1]

# find initial components
numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
	thresh, CONNECTIVITY, cv2.CV_32S)

# remove small components
for label in range(numLabels):
    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    w = stats[label, cv2.CC_STAT_WIDTH]
    h = stats[label, cv2.CC_STAT_HEIGHT]

    neighbor_color = thresh[y - 1][x - 1]

    if w < THRESHOLD or h < THRESHOLD:
        for c in range(x, x + w):
            for r in range(y, y + h):
                if labels[r][c] == label:
                    thresh[r][c] = neighbor_color

# find the components outlines
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# initialize the distances array and draw contours on it (distance on contour is zero)
distances = np.full(thresh.shape, np.inf)
cv2.drawContours(distances, contours[1:], -1, 0, 1) # ASSUMES first contour is always the corners of the image

def minOfSurroundingCube(row, col):
    box = [
        (row - 1, col),
        (row, col + 1),
        (row + 1, col),
        (row, col - 1),
        (row - 1, col - 1),
        (row - 1, col + 1),
        (row + 1, col - 1),
        (row + 1, col + 1),
    ]

    candidates = [(y, x) for (y, x) in box if y >= 0 and y < HEIGHT and x >= 0 and x < WIDTH and distances[y][x] >= 0]

    if len(candidates) == 0:
        return np.inf

    return np.min([distances[y][x] + np.sqrt((y - row) ** 2 + (x - col) ** 2) for (y, x) in candidates])

def maxOfSurroundingCube(row, col):
    box = [
        (row - 1, col),
        (row, col + 1),
        (row + 1, col),
        (row, col - 1),
        (row - 1, col - 1),
        (row - 1, col + 1),
        (row + 1, col - 1),
        (row + 1, col + 1),
    ]

    candidates = [(y, x) for (y, x) in box if y >= 0 and y < HEIGHT and x >= 0 and x < WIDTH and distances[y][x] <= 0]

    if len(candidates) == 0:
        return -np.inf

    return np.max([distances[y][x] - np.sqrt((y - row) ** 2 + (x - col) ** 2) for (y, x) in candidates])

def calculateDistance(y, x):
    if distances[y][x] != 0:
        if TRANSFORM_TYPE == 'I' and thresh[y][x] == 0:
            distances[y][x] = minOfSurroundingCube(y, x)
        elif TRANSFORM_TYPE == 'O' and thresh[y][x] > 0:
            distances[y][x] = minOfSurroundingCube(y, x)
        elif TRANSFORM_TYPE == 'S':
            if thresh[y][x] == 0:
                distances[y][x] = minOfSurroundingCube(y, x)
            else:
                distances[y][x] = maxOfSurroundingCube(y, x)

# forward pass
for y in range(HEIGHT):
    for x in range(WIDTH):
        calculateDistance(y, x)

# backward pass
for y in range(HEIGHT - 1, -1, -1):
    for x in range(WIDTH - 1, -1, -1):
        calculateDistance(y, x)


# normalize
distances[np.isinf(distances)] = 0
output = (((distances - np.min(distances)) / (np.max(distances) - np.min(distances))) * 255).astype(np.uint8)

if OUTPUT_FILE:
    cv2.imwrite(OUTPUT_FILE, output) 

cv2.imshow("Original Image --> Distance Transform", np.hstack((img, cv2.cvtColor(output ,cv2.COLOR_GRAY2RGB))))
cv2.waitKey(0)

