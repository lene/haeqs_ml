from os.path import isfile
from argparse import ArgumentParser
from random import choice

from keras.applications.inception_v3 import InceptionV3

from data_sets.image_file_data_sets import ImageFileDataSets, nth_index_and_value

SIZE = 299
PICKLE_FILE = 'images.{}.pickle.gz'.format(SIZE)
WEIGHTS_FILE = 'images.{}.hdf5'.format(SIZE)


def maxindex(l):
    return nth_index_and_value(l, 1)[0]


def show_image(image, depth):
    data.show_image(image.reshape(*data.size, depth))


def show_demo(num_demo_images):
    for i in range(0, num_demo_images):

        if args.random_demo_images:
            image_index = choice(range(len(data.test.input)))
        else:
            image_index = i

        try:
            show_image(data.test.input[image_index], 3)
        except Exception as e:
            print(str(e))

        actual = data.test.labels[image_index]
        print('actual:', maxindex(actual), data.get_label(maxindex(actual)))

        input = data.test.input[image_index:image_index + 1].reshape(
            1, args.image_size, args.image_size, 3
        )

        for prediction in model.predict(input):
            print('predicted:', *data.prediction_info(prediction, 1))
            print('runner up:', *data.prediction_info(prediction, 2))


def parse_args():
    parser = ArgumentParser(
        description="Train a neural network for image recognition using the Inception V3 geometry."
    )
    parser.add_argument(
        '--image-directory', '-i',
        help='Directory containing the images. The images are stored in subdirectories of this directory. The names of the subdirectories will be used as labels for the images.'
    )
    parser.add_argument(
        '--data-file', '-d', default=PICKLE_FILE,
        help='Gzipped pickle file containing the precomputed data set.'
    )
    parser.add_argument(
        '--weights-file', '-w', default=WEIGHTS_FILE,
        help='HDF5 file containing a precomputed set of weights for this neural network.'
    )
    parser.add_argument(
        '--optimizer', '-o', default='sgd',
        help='Optimizer to use (see http://keras.io/optimizers for valid choices).'
    )
    parser.add_argument(
        '--num-epochs', '-n', type=int, default=1,
        help='How many times to iterate.'
    )
    parser.add_argument(
        '--image-size', '-s', type=int, default=SIZE,
        help='Size (both width and height) to which images are resized.'
    )
    parser.add_argument(
        '--run-test', action='store_true',
        help='Evaluate performance of the model on the test set.'
    )
    parser.add_argument(
        '--num-demo-images', type=int, default=5,
        help='Number of demo images to show and predict.'
    )
    parser.add_argument(
        '--random-demo-images', action='store_true',
        help='Select random images from the test set for demo.'
    )
    parser.add_argument(
        '--predict-folder', help='Folder containing images to run prediction on'
    )
    return parser.parse_args()

args = parse_args()

data = ImageFileDataSets.get_data(args.data_file, args.image_directory, args.image_size)

model = InceptionV3(weights=None, input_shape=(*data.size, data.depth), classes=data.num_labels)

model.compile(loss="categorical_crossentropy", optimizer=args.optimizer, metrics=['accuracy'])

train = data.train.input.reshape(len(data.train.input), args.image_size, args.image_size, 3)

if isfile(args.weights_file):
    print('Loading ' + args.weights_file)
    model.load_weights(args.weights_file)

if args.num_epochs > 0:
    model.fit(train, data.train.labels, epochs=args.num_epochs)
    model.save_weights(args.weights_file)

if args.run_test:
    test = data.test.input.reshape(len(data.test.input), args.image_size, args.image_size, 3)
    loss_and_metrics = model.evaluate(test, data.test.labels)
    print()
    print('test set loss:', loss_and_metrics[0], 'test set accuracy:', loss_and_metrics[1])


if args.num_demo_images:
    show_demo(args.num_demo_images)


def downscale(image, size):
        w, h = image.size
        if w > h:
            image = image.crop(((w-h)/2, 0, w-(w-h)/2, h))
        elif h > w:
            image = image.crop((0, 0, w, w))
        return image.resize(size, Image.BICUBIC)


def normalize(ndarray):
    """Transform a ndarray that contains uint8 values to floats between 0. and 1.

    :param ndarray:
    :return:
    """
    assert isinstance(ndarray, numpy.ndarray)
    assert ndarray.dtype == numpy.uint8

    return numpy.multiply(ndarray.astype(numpy.float32), 1.0/255.0)


if args.predict_folder:
    from os import walk
    from os.path import join, basename
    from PIL import Image
    import numpy
    images, filenames = [], []
    for root, _, files in list(walk(args.predict_folder)):
        for file in files:
            try:
                image = Image.open(join(root, file)).convert('RGB')
            except OSError:
                continue

            images.append(numpy.asarray(downscale(image, (args.image_size, args.image_size))))
            filenames.append((basename(root), file))

    images = normalize(numpy.asarray(images))

    for image_index, prediction in enumerate(model.predict(images)):
            print(filenames[image_index])
            print('predicted:', *data.prediction_info(prediction, 1))
            print('runner up:', *data.prediction_info(prediction, 2))
            # show_image(images[image_index], 3)
            print()

