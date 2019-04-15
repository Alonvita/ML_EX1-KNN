import sys
import math
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt


class CentroidObject:
    def __init__(self, centroid):
        self._value = centroid
        self._nearest_vectors_vector = []

    def get_value(self):
        """
        get_value(self)

        :return: _value
        """
        return self._value

    def add(self, vector):
        """
        add(self, vector)

        :param vector: a vector
        :return: adds the vector to the nearest_vectors_vector
        """
        self._nearest_vectors_vector.append(vector)

    def __str__(self):
        """
        __str__(self)

        :return: a string representation of the _centroid
        """
        return str([", ".join(str("%.2f" % x) for x in self._value)]).replace("0.00", "0.")

    def update(self, heuristics_func):
        # use the given heuristics to update the _centroid
        self._value = heuristics_func(self._nearest_vectors_vector)

        # reinitialize the nearest_vectors_vector
        self._nearest_vectors_vector = []


def init_centroids(X, K):
    """
    Initializes K centroids that are to be used in K-Means on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
    """
    if K == 2:
        return np.asarray([[0.        , 0.        , 0.        ],
                           [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                           [0.49019608, 0.41960784, 0.33333333],
                           [0.02745098, 0.        , 0.        ],
                           [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                           [0.14509804, 0.12156863, 0.12941176],
                           [0.4745098, 0.40784314, 0.32941176],
                           [0.00784314, 0.00392157, 0.02745098],
                           [0.50588235, 0.43529412, 0.34117647],
                           [0.09411765, 0.09019608, 0.11372549],
                           [0.54509804, 0.45882353, 0.36470588],
                           [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                           [0.4745098, 0.38039216, 0.33333333],
                           [0.65882353, 0.57647059, 0.49411765],
                           [0.08235294, 0.07843137, 0.10196078],
                           [0.06666667, 0.03529412, 0.02352941],
                           [0.08235294, 0.07843137, 0.09803922],
                           [0.0745098, 0.07058824, 0.09411765],
                           [0.01960784, 0.01960784, 0.02745098],
                           [0.00784314, 0.00784314, 0.01568627],
                           [0.8627451, 0.78039216, 0.69803922],
                           [0.60784314, 0.52156863, 0.42745098],
                           [0.01960784, 0.01176471, 0.02352941],
                           [0.78431373, 0.69803922, 0.60392157],
                           [0.30196078, 0.21568627, 0.1254902],
                           [0.30588235, 0.2627451, 0.24705882],
                           [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None


def load():
    """
    load_img_file().
    Loads and returns the reshaped image as an array of arrays of RGB values.

    :return: the reshaped image, given the defined sizes.
    """
    # data preparation (loading, normalizing, reshaping)
    img_path = 'dog.jpeg'

    img_read = imread(img_path)
    img_read = img_read.astype(float) / 255.
    img_size = img_read.shape

    return img_read.reshape(img_size[0] * img_size[1], img_size[2])


def euclidean_distance(vector1, vector2):
    """
    euclidean_distance(instance1, instance2).

    :param vector1: a vector on vector_size dimensions
    :param vector2: a vector on vector_size dimensions

    :return: sqrt of the sum of distances between points in every dimension
    """

    return math.sqrt(pow((vector1[0] - vector2[0]), 2) +
                     pow((vector1[1] - vector2[1]), 2) +
                     pow((vector1[2] - vector2[2]), 2))


def find_nearest_vectors_for_centroids(centroids_as_objects, training_set):
    """
    find_nearest_vectors_for_centroids(centroids_as_objects, training_set).

    the function will iterate over the training_set, and use the strategy to calculate the argmin
    of every vector. The vector will be added to the centroid from which armin is minimal.


    :param centroids_as_objects: an array of CentroidsObjects
    :param training_set: the training set
    """

    for vector in training_set:
        minimal_argmin = sys.maxsize
        min_centroid = None

        for centroid in centroids_as_objects:
            argmin = euclidean_distance(centroid.get_value(), vector)

            if argmin < minimal_argmin:
                minimal_argmin = argmin
                min_centroid = centroid

        # after the minimal centroid was found, add the vector to it
        if min_centroid:
            min_centroid.add_vector(vector)


def min_distance_avg_calculated_loss(centroids_as_objects, training_set):
    """
    min_distance_avg_calculated_loss(centroids_as_objects, training_set).

    returns the average of the sum of the minimum distances of all training_set from centroids.

    :param centroids_as_objects:
    :param training_set:
    :return:
    """
    loss_sum = 0

    for vector in training_set:
        min_dist = sys.maxsize

        for centroid in centroids_as_objects:
            distance = euclidean_distance(centroid.get_value(), vector)

            if distance < min_dist:
                min_dist = distance

        loss_sum += min_dist

    return loss_sum / len(training_set)


def vectors_matrix_avg(vectors_matrix):
    """
    vectors_vector_average(vectors_vector)

    :param vectors_matrix:
    :return:
    """
    sum_for_dimension = []

    # append an empty list for each column in the matrix
    for dim in range(len(vectors_matrix[0])):
        sum_for_dimension.append([])

    # sum columns
    for vector in vectors_matrix:
        for dim in range(len(vector)):
            sum_for_dimension[dim].append(vector[dim])

    averages_arr = []

    # avg the arrays created
    for arr in sum_for_dimension:
        arr_sum = 0

        for value in arr:
            arr_sum += value

        arr_sum = arr_sum / len(arr)

        averages_arr.append(arr_sum)

    return averages_arr


def print_iteration(iteration, centroids):
    sys.stdout.write("iter " + str(iteration) + ": ")

    for x in range(len(centroids) - 1):
        sys.stdout.write(str(centroids[x]) + ", ")

    print(str(centroids[len(centroids) - 1]))


def knn(raw_centroids, training_set):
    k = len(raw_centroids)
    loss_vector = []
    generated_centroids = []

    for c in raw_centroids:
        generated_centroids.append(CentroidObject(c))

    print_iteration(0, generated_centroids)

    for it in range(1, 10):
        find_nearest_vectors_for_centroids(generated_centroids, training_set)

        calculated_loss = min_distance_avg_calculated_loss(generated_centroids, training_set)

        loss_vector.append(calculated_loss)

        # update all centroids
        for c in generated_centroids:
            c.update(vectors_matrix_avg)

        print_iteration(it, generated_centroids)

    loss_vector = loss_vector[1:]

    plt.plot(range(len(loss_vector)), loss_vector)
    plt.title('Average loss after 10 iterations for k = %d ' % k)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.savefig('loss_funcs_plots')


def main():
    image = load()

    # run the KNN algorithm for each of the values in the given array
    for k in [2, 4, 8, 16]:
        print("k=" + str(k) + ":")

        # run knn algorithm
        knn(init_centroids(image, k), image)

    plt.clf()


if __name__ == "__main__":
    main()
