from keras.layers import Dense, Activation
from keras.models import Sequential

from data_sets.image_file_data_sets import ImageFileDataSets

data = ImageFileDataSets('/home/lene/Pictures/S', 256, 256, './data', 0, True)

model = Sequential([
    Dense(1024, input_dim=data.num_features), Activation('relu'),
    Dense(1024), Activation('relu'),
    Dense(1024), Activation('relu'),
     Dense(data.num_labels), Activation('softmax'),
])

model.compile(
    optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']
)

model.fit(data.train.input, data.train.labels)

loss_and_metrics = model.evaluate(data.test.input, data.test.labels)
print()
print('test set loss:', loss_and_metrics[0], 'test set accuracy:', loss_and_metrics[1])


def maxindex(l):
    max_value = max(l)
    return list(l).index(max_value)


def nth_index_and_value(l, n):
    s = sorted(l)
    v = s[-n]
    i = list(l).index(v)
    return i, v


def show_image(image, depth):
    data.show_image(image.reshape(*data.size, depth))


for i in range(0, 5):

    show_image(data.test.input[i], 3)

    actual = data.test.labels[i]
    print('actual:', maxindex(actual), data.get_label(maxindex(actual)))

    prediction = model.predict(data.test.input[i:i + 1])[0]
    print(
        'predicted:', maxindex(prediction), data.get_label(maxindex(prediction)), nth_index_and_value(prediction, 1),
        'runner up:', nth_index_and_value(prediction, 2)
    )
