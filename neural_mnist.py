from keras.layers import Dense, Activation
from keras.models import Sequential

from data_sets.mnist_data_sets import MNISTDataSets

data = MNISTDataSets('./data', True)

mnist = Sequential([
    Dense(32, input_dim=data.num_features), Activation('relu'),
    Dense(data.num_labels), Activation('softmax'),
])

mnist.compile(
    optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']
)

mnist.fit(data.train.input, data.train.labels)


def show_image(grayscale_values):
    import matplotlib.pyplot as plt
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
