import gzip
from os import walk
from PIL import Image

import numpy
from .input_data import maybe_download, images_from_bytestream, \
    read_one_image_from_file, read_one_image_from_url, read_images_from_file, read_images_from_url, \
    read_images_from_files, read_images_from_urls

from .data_sets import DataSets
from data_sets.images_labels_data_set import ImagesLabelsDataSet

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

"""DataSets for RGB images read from files using the directory name as label."""


class ImageFileDataSets(DataSets):

    """Data sets (training, validation and test data) containing RGB image files."""
    DEFAULT_VALIDATION_SHARE = 0.2

    def __init__(self, base_dir, x_size, y_size, train_dir, validation_share=None, one_hot=False):
        """Construct the data set, locating and if necessarily downloading the MNIST data.

        :param train_dir: Where to store the MNIST data files.
        :param one_hot:
        """
        self.train_dir = train_dir
        self.one_hot = one_hot
        self.base_dir = base_dir
        self.size = (x_size, y_size)
        self.num_features = x_size*y_size*3

        all_images, all_labels = self._extract_images(base_dir)

        if False:
            for i, image in enumerate(all_images):
                # self.show_image(image, all_labels[i])
                print(image)

        self.num_labels = len(set(all_labels))
        if one_hot:
            all_labels, self.labels_to_numbers = _dense_to_one_hot(all_labels)
            self.numbers_to_labels = {v: k for k, v in self.labels_to_numbers.items()}

        train_images, train_labels, test_images, test_labels = self.split_images(all_images, all_labels, 0.8)

        self.validation_size = int(len(all_images)*(self.DEFAULT_VALIDATION_SHARE if validation_share is None else validation_share))
        validation_images = train_images[:self.validation_size]
        validation_labels = train_labels[:self.validation_size]
        train_images = train_images[self.validation_size:]
        train_labels = train_labels[self.validation_size:]

        super().__init__(
            ImagesLabelsDataSet(train_images, train_labels, 3),
            ImagesLabelsDataSet(validation_images, validation_labels, 3),
            ImagesLabelsDataSet(test_images, test_labels, 3)
        )

    def get_label(self, number):
        return self.numbers_to_labels[number]

    ############################################################################

    def _extract_images(self, base_dir):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        import os.path
        print('Extracting', base_dir)
        images, labels = [], []
        all_dirs = list(walk(base_dir))
        i = 0
        for root, dirs, files in all_dirs:
            label = root.split('/')[-1]
            print(label, "%.2f%%" % (i/len(all_dirs)*100))
            i += 1
            for file in files:
                try:
                    image = Image.open(os.path.join(root, file)).convert('RGB')
                except OSError:
                    continue

                images.append(numpy.asarray(self.downscale(image)))
                labels.append(label)

        return numpy.asarray(images), numpy.asarray(labels)

    def downscale(self, image):
        w, h = image.size
        if w > h:
            image = image.crop(((w-h)/2, 0, w-(w-h)/2, h))
        elif h > w:
            image = image.crop((0, 0, w, w))
        return image.resize(self.size, Image.BICUBIC)

    def show_image(self, rgb_values, label=''):
        import matplotlib.pyplot as plt
        plt.imshow(rgb_values, cmap='gray')
        plt.title(label)
        plt.show()

    def split_images(self, images, labels, train_to_test_ratio):
        from random import shuffle
        test_size = int(len(images)*(1-train_to_test_ratio))
        combined = list(zip(images, labels))
        shuffle(combined)
        images[:], labels[:] = zip(*combined)
        return images[test_size:], labels[test_size:], images[:test_size], labels[:test_size]


def _dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    num_classes = len(set(labels_dense))
    num_labels = labels_dense.shape[0]
    labels_to_numbers = {label: i for i, label in enumerate(list(set(labels_dense)))}
    labels_as_numbers = numpy.asarray([labels_to_numbers[label] for label in labels_dense])

    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_as_numbers.ravel()] = 1
    return labels_one_hot, labels_to_numbers


