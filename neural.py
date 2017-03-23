from keras.models import Sequential
from keras.layers import Dense, Activation

from mnist_data_sets import MNISTDataSets

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


def maxindex(l):
    max_value = max(l)
    return list(l).index(max_value)


for i in range(0, 10):
    print('expected:', maxindex(data.test.labels[i]), data.test.labels[i])
    plt.imshow(MNISTDataSets.to_image_data(data.test.input[i]), cmap='gray')
    plt.show()
    print('predicted:', maxindex(mnist.predict(data.test.input[i:i+1])[0]), mnist.predict(data.test.input[i:i+1])[0])


