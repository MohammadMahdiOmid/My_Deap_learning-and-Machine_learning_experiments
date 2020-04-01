import cv2
import numpy as np


def main():
    # initialize the class labels and set the seed of the pseudorandom
    # number generator so we can reproduce our results
    labels = ["cat", "dog", "panda"]
    np.random.seed(1)

    # randomly initialize our weight matrix and bias vector -- in a
    # *real* training and classification task, these parameters would
    # be *learned* by our model, but for the sake of this example,
    # let’s use random values

    w = np.random.rand(3, 3072)
    b = np.random.rand(3)

    # load our example image, resize it, and then flatten it into our
    # "feature vector" representation

    orig = cv2.imread("animals.jpg")
    image = cv2.resize(orig, (32, 32)).flatten()

    # compute the output scores by taking the dot product between the
    # weight matrix and image pixels, followed by adding in the bias

    scores = w.dot(image) + b

    # loop over the scores + labels and display them
    for (label, score) in zip(labels, scores):
        print("[INFO] {}: {:.2f}".format(label, score))

    # draw the label with the highest score on the image as our
    # prediction
    cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # display our input image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
