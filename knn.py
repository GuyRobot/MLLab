import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn import metrics


def euclidean_distance(point1, point2):
    # calculating Euclidean distance
    # using linalg.norm()
    dist = np.linalg.norm(point1 - point2)
    return dist


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append(dist)
    distances = np.array(distances)
    neighbors = train[np.argpartition(distances, num_neighbors)[:num_neighbors]]
    return neighbors


def predict(train, test, k):
    labels = []
    for item in test:
        neighbors = get_neighbors(train, item, k)
        pred_labels = neighbors[:, -1]
        # noinspection PyUnresolvedReferences
        labels.append(mode(pred_labels).mode[0])
    return labels


def report(train_f, test_f, k):
    train_set = pd.read_csv(train_f, sep='[,,\s]', header=None, engine='python')
    test_set = pd.read_csv(test_f, sep='[,,\s]', header=None, engine='python')

    x_train = train_set.values  # Get training data points (exclude class value)

    num_row, num_col = train_set.shape
    test_num_row, test_num_col = test_set.shape

    x_test = test_set.values  # Get training data points (exclude class value)
    y_test = test_set.iloc[:, test_set.shape[1] - 1].values  # Get training class data points (the last column)

    pred = predict(x_train, x_test, k)

    print("-" * 24 + "CLASSIFY RESULT WITH K = %s" % k + "-" * 24)
    print("⁕ TRAIN FILE: %s, WITH %d SAMPLES" % (train_f, num_row))
    print("⁕ TEST FILE: %s, WITH %d SAMPLES" % (test_f, test_num_row))
    print("⁕ ACCURACY SCORE: %d%%" % (metrics.accuracy_score(y_test, pred) * 100))
    print("⁕ CONFUSION MATRIX:\n", metrics.confusion_matrix(y_test, pred))
    print("⁕ CLASSIFICATION REPORT:\n", metrics.classification_report(y_test, pred))
    print("-" * 24 + "END OF CLASSIFY RESULT WITH K = %s" % k + "-" * 24)


if __name__ == '__main__':
    report('data/faces/data.trn', 'data/faces/data.tst', 1)
