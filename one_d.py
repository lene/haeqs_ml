from __future__ import print_function
from math import sqrt
import sys

# to use matplotlib to plot values:
# sudo apt install python3-tk
# pip install matplotlib
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]  # example values for x
y = [1, 3, 3, 2, 5]  # example values

plt.plot(y, 'ro')
plt.show()
sys.exit()


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def variance(numbers, mean):
    return sum((x - mean) ** 2 for x in numbers)


def covariance(x, mean_x, y, mean_y):
    return sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))


def coefficients(x, y):
    x_m, y_m = mean(x), mean(y)
    b1 = covariance(x, x_m, y, y_m) / variance(x, x_m)
    b0 = y_m - b1 * x_m
    return b0, b1


coeffs = coefficients(x, y)
print()
print("Coefficients: {}\ny = {}*x + {}".format(coeffs, coeffs[0], coeffs[1]))


def prediction(training_x, training_y, x):
    b0, b1 = coefficients(training_x, training_y)
    return b0 + b1 * x


print()
print("Prediction for x = 1: {}".format(prediction(x, y, 1)))
print("Actual value for x = 1: {}".format(y[0]))
print("Prediction for x = 2: {}".format(prediction(x, y, 2)))
print("Actual value for x = 2: {}".format(y[1]))
# ...
print("Prediction for x = 5: {}".format(prediction(x, y, 5)))
print("Actual value for x = 5: {}".format(y[4]))

plt.plot([prediction(x, y, xx) for xx in range(1, 6)])
# plt.show()


def simple_linear_regression(train_x, train_y, test_x):
    return [prediction(train_x, train_y, test_x[i]) for i in range(len(test_x))]


def rmse(actual, predicted):
    errors = [(predicted[i] - actual[i]) ** 2 for i in range(len(actual))]
    mean_error = sum(errors) / float(len(actual))
    return sqrt(mean_error)


def test_on_test_set(algorithm, dataset):
    test_set = list()
    for row in dataset:
        row_copy = list(row)
        test_set.append(row_copy)
    test_set = dataset[0][:-2]
    predicted = algorithm(dataset[0], dataset[1], test_set)
    return predicted, test_set


def evaluate_algorithm(dataset, algorithm):
    predicted, test_set = test_on_test_set(algorithm, dataset)
    actual = [row[-1] for row in dataset]
    return rmse(actual, predicted)


predicted, test_set = test_on_test_set(simple_linear_regression, (x, y))
print()
print("Values:", test_set, "Prediction:", predicted)
print("Error:", evaluate_algorithm((x, y), simple_linear_regression))


def load_csv(filename):
    from csv import reader
    x, y = [], []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            if not row[0].isdigit():
                continue
            if not int(row[0]):
                continue
            x.append(int(row[0]))
            y.append(float(row[1]))
    return x, y

x, y = load_csv('Demographic_Statistics_By_Zip_Code.csv')
print(x, y)
plt.clf()
plt.plot(x, y, 'ro')
# plt.show()
coeffs = coefficients(x, y)
print("Coefficients: {}\ny = {}*x + {}".format(coeffs, coeffs[0], coeffs[1]))
print("Error:", evaluate_algorithm((x, y), simple_linear_regression))

plt.plot([prediction(x, y, xx) for xx in range(0, 250)])
# plt.show()
