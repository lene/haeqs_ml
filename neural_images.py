from os.path import isfile
from pickle import dump, load
from gzip import open as gzopen
from keras.applications.inception_v3 import InceptionV3

from data_sets.image_file_data_sets import ImageFileDataSets

SIZE = 299
PICKLE_FILE = 'images.{}.pickle.gz'.format(SIZE)


def get_data():
    global data
    if isfile(PICKLE_FILE):
        with gzopen(PICKLE_FILE, 'rb') as file:
            return load(file)
    else:
        data = ImageFileDataSets('/home/lene/Pictures/S', SIZE, SIZE, './data', 0, True)
        with gzopen(PICKLE_FILE, 'wb') as file:
            dump(data, file)
        return data

data = get_data()

model = InceptionV3(weights=None, include_top=True, input_shape=(SIZE, SIZE, 3), classes=data.num_labels)

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])

input = data.train.input.reshape(len(data.train.input), SIZE, SIZE, 3)

if isfile('images.hdf5'):
    model.load_weights('images.hdf5')

model.fit(input, data.train.labels, epochs=5)
model.save_weights('images.hdf5')

input = data.test.input.reshape(len(data.test.input), SIZE, SIZE, 3)
loss_and_metrics = model.evaluate(input, data.test.labels)
print()
print('test set loss:', loss_and_metrics[0], 'test set accuracy:', loss_and_metrics[1])



def maxindex(l):
    max_value = max(l)
    return list(l).index(max_value)


def nth_index_and_value(l, n):
    v = sorted(l)[-n]
    i = list(l).index(v)
    return i, v


def show_image(image, depth):
    data.show_image(image.reshape(*data.size, depth))


from keras.applications.imagenet_utils import decode_predictions

for i in range(0, 5):

    show_image(data.test.input[i], 3)

    actual = data.test.labels[i]
    print('actual:', maxindex(actual), data.get_label(maxindex(actual)))

    input = data.test.input[i:i + 1].reshape(1, SIZE, SIZE, 3)

    prediction = model.predict(input)
    print(prediction)
    print(decode_predictions(prediction))
    # print(
    #     'predicted:', maxindex(prediction), data.get_label(maxindex(prediction)), nth_index_and_value(prediction, 1),
    #     'runner up:', nth_index_and_value(prediction, 2)
    # )
