from skimage.exposure import rescale_intensity
import numpy as np
import cv2


def convolve(image, K):
    # for grab dimentions of image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # we use below codes for have a same dimention of input and output
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    # lop for input image for "sliding" the kernel cross each(x,y)coordinate from left to right and top to bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # this part is for extract Region Of Interest:
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # perform the actual convolution
            K = (roi * K).sum()

            # store the convolved value in the output (x, y)-
            # coordinate of the output image
            output[y - pad, x - pad] = K

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output


# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

# construct an emboss kernel
emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype="int")

# construct the kernel bank, a list of kernels we’re going to apply
# using both our custom ‘convole‘ function and OpenCV’s ‘filter2D‘
# function
kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY),
    ("emboss", emboss))

image = cv2.imread("dog.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (kernelName, K) in kernelBank:
    # apply the kernel to the grayscale image using both our custom
    # ‘convolve‘ function and OpenCV’s ‘filter2D‘ function
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

# show the output images
cv2.imshow("Original", gray)
cv2.imshow("{} - convole".format(kernelName), convolveOutput)
cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
