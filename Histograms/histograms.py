import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys


def calcHistogram(img):
    hist = np.zeros((256))

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            hist[img[y][x]] += 1

    return hist


def normalizeHistogram(hist, dim):
    return hist / (dim[0] * dim[1])


def getEqualizationTranform(norm):
    T = np.zeros(256, dtype=np.uint8)

    sum = 0
    for ri in range(256):
        sum += norm[ri]
        T[ri] = np.round(255 * sum)

    return T


def equalizeHistogram(img, norm):
    T = getEqualizationTranform(norm)
    return np.vectorize(lambda x: T[x])(img)


def matchHistogram(img1, img2):
    norm1 = normalizeHistogram(calcHistogram(img1), img1.shape)
    norm2 = normalizeHistogram(calcHistogram(img2), img2.shape)

    T1 = getEqualizationTranform(norm1)
    T2 = getEqualizationTranform(norm2)

    T2i = np.zeros(256, dtype=np.uint8)

    seqStart = 1
    for i in range(1, 256):
        if T2[i] == T2[i - 1]:
            continue
        else:
            T2i[T2[i]] = i

            if i - seqStart > 0:
                step = (T2[i] - T2[seqStart - 1]) / (i - seqStart)
                for j in range(seqStart, i):
                    T2[j] = min(int(round(T2[j - 1] + step)), 255)

            seqStart = i

    return np.vectorize(lambda x: T2i[T1[x]])(img1)


def plotHist(fig, hist):
    fig.hist(range(len(hist)), bins=len(hist), weights=hist)
    fig.set_xlabel("Intensity")
    fig.set_ylabel("Probability")


OP = sys.argv[1]
IMG_SRC = sys.argv[2]
MATCH_SRC = sys.argv[3] if len(sys.argv) > 3 else None
OUTPUT = sys.argv[4] if len(sys.argv) > 4 else None

img = cv2.imread(IMG_SRC, cv2.IMREAD_GRAYSCALE)

hist = calcHistogram(img)
norm = normalizeHistogram(hist, img.shape)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# display input image
axs[0][0].set_title("Input Image")
axs[0][0].imshow(img, cmap="gray")

# display input histogram
# axs[0][1].set_title("Input Histogram")
# plotHist(axs[0][1], norm)

if OP == "-H":
    pass
if OP == "-E":
    equalized_img = equalizeHistogram(img, norm)
    axs[1][0].set_title("Equalized Image")
    axs[1][0].imshow(equalized_img, cmap="gray")

    equalized_hist = normalizeHistogram(
        calcHistogram(equalized_img), equalized_img.shape)
    axs[1][1].set_title("Equalized Histogram")
    plotHist(axs[1][1], equalized_hist)
elif OP == "-M":
    match_img = cv2.imread(MATCH_SRC, cv2.IMREAD_GRAYSCALE)

    # display image 2 histogram
    axs[0][1].set_title("Input2 Histogram")
    plotHist(axs[0][1], normalizeHistogram(
        calcHistogram(match_img), match_img.shape))

    matched_img = matchHistogram(img, match_img)
    axs[1][0].set_title("Matched Image")
    axs[1][0].imshow(matched_img, cmap="gray")

    matched_hist = normalizeHistogram(
        calcHistogram(matched_img), matched_img.shape)
    axs[1][1].set_title("Matched Histogram")
    plotHist(axs[1][1], matched_hist)


plt.tight_layout()
plt.show()
