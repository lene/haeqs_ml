from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from os.path import isfile

from data_sets.mnist_data_sets import MNISTDataSets

def larger_model():
        # http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
	# create model
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
    
data = MNISTDataSets('./data', True)

mnist = Sequential([
    # Dense(32, input_dim=data.num_features), Activation('relu'),
    Dense(data.num_features, input_dim=data.num_features), Activation('relu'),
    Dense(data.num_labels), Activation('softmax'),
])
# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
# fancy_mnist = Sequential([
#     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(data.num_labels, activation='softmax')
# ])

mnist.compile(
    optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']
)
# fancy_mnist.compile(
#     optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['accuracy']
# )
if isfile('mnist.hdf5'):
    mnist.load_weights('mnist.hdf5')
else:
    mnist.fit(data.train.input, data.train.labels)
    # fancy_mnist.fit(data.train.input.reshape(data.train.input.shape[0], 28, 28, 1), data.train.labels)
    mnist.save_weights('mnist.hdf5')

loss_and_metrics = mnist.evaluate(data.test.input, data.test.labels)
print()
print('test set loss:', loss_and_metrics[0], 'test set accuracy:', loss_and_metrics[1])

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
