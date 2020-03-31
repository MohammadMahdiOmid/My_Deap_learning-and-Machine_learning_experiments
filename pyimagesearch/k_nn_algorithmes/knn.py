from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessors.image_preprocessor import ResizePreprocessor
from datasets.simpleDatasetLoader import SimpleDatasetLoader
from imutils import paths


def main():
    k = 2
    j = 2

    # construct the argument parse and parse the arguments

    # grab the list of images that we’ll be describing
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images('/home/mahdi/Downloads/Telegram Desktop/cats_and_dogs'))

    # initialize the image preprocessors, load the dataset from disk,
    # and reshape the data matrix
    sp = ResizePreprocessor(32, 32)
    sdl = SimpleDatasetLoader(preprocessor=[sp])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.reshape((data.shape[0], 3072))

    # show some information on memory consumption of the images
    print("[INFO] features matrix: {:.1f}MB".format(
        data.nbytes / (1024 * 1000.0)))

    # encode the labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing

    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.25, random_state=42)

    # train and evaluate a k-NN classifier on the raw pixel intensities
    print("[INFO] evaluating k-NN classifier...")
    model = KNeighborsClassifier(n_neighbors=k,
                                 n_jobs=j)
    model.fit(trainX, trainY)
    print(classification_report(testY, model.predict(testX),
                                target_names=le.classes_))


if __name__ == '__main__':
    main()
