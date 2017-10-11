from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

from data_sets.mnist_data_sets import MNISTDataSets


def very_simple_mnist():
    return Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(32, activation='relu'),
        Dense(data.num_labels, activation='softmax')
    ])


def simple_mnist():
    return Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(data.num_features, input_dim=data.num_features, activation='relu'),
        Dense(data.num_labels, activation='softmax'),
    ])


def fancy_mnist():
    from keras.layers import Dropout, Conv2D, MaxPooling2D
    # https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    return Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(data.num_labels, activation='softmax')
    ])


def even_fancier_mnist():
    from keras.layers import Dropout, Conv2D, MaxPooling2D
    # http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
    model = Sequential([
        Conv2D(30, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(15, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(50, activation='relu'),
        Dense(data.num_labels, activation='softmax')
    ])
    # Compile model
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def show_image(grayscale_values):
    import matplotlib.pyplot as plt
    plt.switch_backend('TkAgg')
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

data = MNISTDataSets('./data', True)

mnist = very_simple_mnist()
# mnist = simple_mnist()
# mnist = fancy_mnist()
# mnist = even_fancier_mnist()

mnist.compile(
    optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']
)

# simple models:
# mnist.fit(data.train.input, data.train.labels, epochs=10)
# loss_and_metrics = mnist.evaluate(data.test.input, data.test.labels)
# fancy models:
mnist.fit(data.train.input.reshape(55000, 28, 28, 1), data.train.labels, epochs=1)
loss_and_metrics = mnist.evaluate(data.test.input.reshape(10000, 28, 28, 1), data.test.labels)

print()
print('test set loss:', loss_and_metrics[0], 'test set accuracy:', loss_and_metrics[1])

from keras.utils import plot_model
from subprocess import call

plot_model(mnist, show_shapes=True, to_file='model.png')
call(['xv', 'model.png'])


for i in range(0, 10):
    actual = data.test.labels[i]
    print('actual:   ', maxindex(actual), actual)

    show_image(data.test.input[i])

    # prediction = mnist.predict(data.test.input[i:i + 1])[0]
    prediction = mnist.predict(data.test.input[i:i + 1].reshape(1, 28, 28, 1))[0]
    print('predicted:', maxindex(prediction), nth_index_and_value(prediction, 1), 'runner up:', nth_index_and_value(prediction, 2))
