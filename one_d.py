from math import sqrt

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def variance(numbers, avg):
    return sum((x - avg) ** 2 for x in numbers)


def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


def coefficients(x, y):
    b1 = covariance(x, mean(x), y, mean(y)) / variance(x, mean(x))
    b0 = mean(y) - b1 * mean(x)
    return b0, b1


def prediction(training_x, training_y, x):
    b0, b1 = coefficients(training_x, training_y)
    return b0 + b1 * x


def simple_linear_regression(train_x, train_y, test_x):
    return [prediction(train_x, train_y, test_x[i]) for i in range(len(test_x))]


def rmse(actual, predicted):
    errors = [(predicted[i] - actual[i]) ** 2 for i in range(len(actual))]
    mean_error = sum(errors) / float(len(actual))
    return sqrt(mean_error)


def evaluate_algorithm(dataset, algorithm):
    test_set = list()
    for row in dataset:
        row_copy = list(row)
        # row_copy[-1] = None
        test_set.append(row_copy)
    test_set = dataset[0][:-2]
    predicted = algorithm(dataset[0], dataset[1], test_set)
    print(predicted)
    actual = [row[-1] for row in dataset]
    return rmse(actual, predicted)

x = [1, 2, 4, 3, 5]
y = [1, 3, 3, 2, 5]

evaluate_algorithm((x, y), simple_linear_regression)



