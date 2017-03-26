from os.path import isfile
from subprocess import call
from pickle import dump, load
from gzip import open as gzopen
from argparse import ArgumentParser
from random import choice

from keras.applications.inception_v3 import InceptionV3

from data_sets.image_file_data_sets import ImageFileDataSets, nth_index_and_value

SIZE = 299
PICKLE_FILE = 'images.{}.pickle.gz'.format(SIZE)
WEIGHTS_FILE = 'images.{}.hdf5'.format(SIZE)


def get_data(args):
    if isfile(args.data_file):
        print('Loading ' + args.data_file)
        with gzopen(args.data_file, 'rb') as file:
            return load(file)
    else:
        data = ImageFileDataSets(args.image_directory, args.image_size, args.image_size, 0, True)
        try:
            with gzopen(args.data_file, 'wb') as file:
                dump(data, file)
        except OverflowError:   # annoying python bug when using gzopen with data > 4GB
            uncompressed_file = '.'.join(args.data_file.split('.')[:-1])
            with open(uncompressed_file, 'wb') as file:
                dump(data, file)
            call(('gzip', uncompressed_file))
        return data


def maxindex(l):
    return nth_index_and_value(l, 1)[0]


def show_image(image, depth):
    data.show_image(image.reshape(*data.size, depth))


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
    return parser.parse_args()

args = parse_args()
data = get_data(args)

model = InceptionV3(weights=None, include_top=True, input_shape=(args.image_size, args.image_size, 3), classes=data.num_labels)

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


for i in range(0, args.num_demo_images):

    if args.random_demo_images:
        image_index = choice(range(len(data.test.input)))
    else:
        image_index = i

    show_image(data.test.input[image_index], 3)

    actual = data.test.labels[image_index]
    print('actual:', maxindex(actual), data.get_label(maxindex(actual)))

    input = data.test.input[image_index:image_index + 1].reshape(
        1, args.image_size, args.image_size, 3
    )

    for prediction in model.predict(input):
        print('predicted:', *data.prediction_info(prediction, 1))
        print('runner up:', *data.prediction_info(prediction, 2))
