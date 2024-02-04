import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt


def pad_matrix(matrix, n):
    copy = np.zeros(
        (matrix.shape[0] + 2 * n, matrix.shape[1] + 2 * n), dtype=np.uint8)
    copy[n:-n, n:-n] = image

    copy[:n, n:-n] = image[0, :]
    copy[-n:, n:-n] = image[-1, :]
    copy[n:-n, :n] = image[:, 0].reshape((-1, 1))
    copy[n:-n, -n:] = image[:, -1].reshape((-1, 1))

    copy[0, 0] = copy[0, 1]
    copy[0, -1] = copy[0, -2]
    copy[-1, 0] = copy[-1, 1]
    copy[-1, -1] = copy[-1, -2]

    return copy


def apply_filter(image, fil):
    n = fil.shape[0] // 2

    # Pad the input image as preperation for convolution
    copy = pad_matrix(image, n)

    output = np.zeros(image.shape, dtype=np.float32)
    # Perform convolution
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            start_y, start_x = y, x
            end_y, end_x = y + fil.shape[0], x + fil.shape[0]
            output[y, x] = (fil * copy[start_y: end_y, start_x: end_x]).sum()

    # normalize output
    output = ((output - output.min()) /
              (output.max() - output.min()) * 255).round()

    return output


def draw_filter(ax, fil):
    ax.set_title('Filter used (denormalized)')
    ax.matshow(fil, cmap='viridis')
    for i in range(len(fil)):
        for j in range(len(fil)):
            c = fil[j, i]
            ax.text(i, j, str(c), va='center', ha='center')


def gaussian_filter(image, size, sigma):
    if size % 2 == 0:
        raise ValueError("Size must be an odd number")

    center = size // 2

    fil = np.zeros((size, size), dtype=np.float32)

    def gaussian(x, y):
        return np.exp(- (x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

    for y in range(size):
        for x in range(size):
            fil[y, x] = gaussian(x - center, y - center)

    output = apply_filter(image, fil)

    return output, (fil / fil.min()).round().astype(np.int32)


def edge_detect(image, size, direction):
    if direction == 0:
        fil = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    else:
        fil = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    output = apply_filter(image, fil)
    _, output = cv2.threshold(output, 150, 255, cv2.THRESH_BINARY)

    return output, fil


def selection_filter(image, size, type):
    func = None
    if type == 'min':
        func = np.min
    elif type == 'max':
        func = np.max
    elif type == 'median':
        func = np.median
    else:
        raise ValueError("Invalid type")

    copy = pad_matrix(image, size // 2)

    output = np.zeros(image.shape, dtype=np.float32)

    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            start_y, start_x = y, x
            end_y, end_x = y + size, x + size
            output[y, x] = func(copy[start_y: end_y, start_x: end_x])

    return output


if len(sys.argv) != 6 and len(sys.argv) != 8:
    print(
        "Usage: python3 filter.py -G/E/S <dimension> -p <params> <input> [-o <output>]")
    sys.exit(1)

MODE = sys.argv[1][1]
SIZE = int(sys.argv[2])
PARAM = sys.argv[sys.argv.index('-p') + 1]
IMAGE_PATH = sys.argv[sys.argv.index('-p') + 2]
OUTPUT_PATH = sys.argv[sys.argv.index(
    '-o') + 1] if '-o' in sys.argv else None

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].set_title('Original')
ax[0].imshow(image, cmap='gray')

if MODE == 'G':
    output, fil = gaussian_filter(image, SIZE, float(PARAM))
    ax[1].set_title('Output')
    ax[1].imshow(output, cmap='gray')

    draw_filter(ax[2], fil)
elif MODE == 'E':
    output, fil = edge_detect(image, SIZE, float(PARAM))
    ax[1].set_title('Output')
    ax[1].imshow(output, cmap='gray')

    draw_filter(ax[2], fil)

elif MODE == 'S':
    output = selection_filter(image, SIZE, PARAM)
    ax[1].set_title('Output')
    ax[1].imshow(output, cmap='gray')

plt.show()
