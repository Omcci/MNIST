import numpy as np
import struct
from array import array


class MnistDataLoader(object):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        # Load training set
        self.X_train, self.y_train = self.read_images_labels(
            training_images_filepath, training_labels_filepath
        )

        # Load test set
        self.X_test, self.y_test = self.read_images_labels(
            test_images_filepath, test_labels_filepath
        )

        # Check shapes
        print("Train images:", np.array(self.X_train).shape)
        print("Train labels:", np.array(self.y_train).shape)
        print("Test images:", np.array(self.X_test).shape)
        print("Test labels:", np.array(self.y_test).shape)

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(rows, cols)
            images[i][:] = img

        return images, labels
