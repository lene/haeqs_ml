from keras.layers import Dense, Activation
from keras.models import Sequential

from data_sets.mnist_data_sets import MNISTDataSets

mnist = Sequential([
    Dense(32, input_dim=784), Activation('relu'),
    Dense(10), Activation('softmax'),
])

mnist.compile(
    optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']
)

data = MNISTDataSets('./data', True)

mnist.fit(data.train.input, data.train.labels)

loss_and_metrics = mnist.evaluate(data.test.input, data.test.labels)
print()
print('test set loss:', loss_and_metrics[0], 'test set accuracy:', loss_and_metrics[1])

import matplotlib.pyplot as plt


def show_image(grayscale_values):
    plt.imshow(to_image_data(grayscale_values), cmap='gray')
    plt.show()


def to_image_data(dataset):
    return dataset.reshape(28, 28)


def maxindex(l):
    max_value = max(l)
    return list(l).index(max_value)


def nth_index_and_value(l, n):
    s = sorted(l)
    v = s[-n]
    i = list(l).index(v)
    return i, v


for i in range(0, 10):
    show_image(data.test.input[i])

    actual = data.test.labels[i]
    print('actual:', maxindex(actual), actual)

    prediction = mnist.predict(data.test.input[i:i + 1])[0]
    print('predicted:', maxindex(prediction), nth_index_and_value(prediction, 1), 'runner up:', nth_index_and_value(prediction, 2))
