"""
mc_perceptron.py

A class for building, training, and getting analytics for a Multi-Class Perceptron machine learning model in Python.

Loads feature data and examples from the feature_data sub-directory, in the predefined format (found in the README),
and builds a mc_perceptron model object, to be saved in the classifier_models directory.
"""
from feature_data.shapes_example import classes, feature_list, feature_data
import numpy as np
import pickle
import random

__author__ = "Sidd Karamcheti"

# Constants
BIAS = 1                            # Dummy Feature for use in setting constant factor in Training.
TRAIN_TEST_RATIO = .75              # Default Ratio of data to be used in Training vs. Testing.
ITERATIONS = 100                    # Default Number of Training Iterations.
OUTPUT_PATH = "classifier_models/"  # Directory in which to save completed models.


class MultiClassPerceptron():
    """
    A Multi-Class Perceptron Model object, with functions for loading feature data, training the algorithm,
    and running analytics on model performance.

    :param  classes           List of categories/classes (match tags in tagged data).
    :param  feature_list      List of features.
    :param  feature_data      Feature Data, in format specified in README, usually imported from feature_data module.
    :param  train_test_ratio  Ratio of data to be used in training vs. testing. Set to 75% by default.
    :param  iterations        Number of iterations to run training data through. Set to 100 by default.
    """
    def __init__(self, classes, feature_list, feature_data, train_test_ratio=TRAIN_TEST_RATIO, iterations=ITERATIONS):
        self.classes = classes
        self.feature_list = feature_list
        self.feature_data = random.shuffle(feature_data)
        self.ratio = train_test_ratio
        self.iterations = iterations

        # Split feature data into train set, and test set
        self.train_set = self.feature_data[:(len(self.feature_data) * self.ratio)]
        self.test_set = self.feature_data[(len(self.feature_data) * self.ratio):]

        # Initialize empty weight vectors, with extra BIAS term.
        self.weight_vectors = {c: np.array([0 for _ in xrange(len(feature_list) + 1)]) for c in self.classes}

    def train(self):
        """
        Train the Multi-Class Perceptron algorithm using the following method (from the README):

        During each iteration of training, the data (formatted as a feature vector) is read in, and the dot
        product is taken with each unique weight vector (which are all initially set to 0). The class that
        yields the highest product is the class to which the data belongs. In the case this class is the
        correct value (matches with the actual category to which the data belongs), nothing happens, and the
        next data point is read in. However, in the case that the predicted value is wrong, the weight vectors a
        re corrected as follows: The feature vector is subtracted from the predicted weight vector, and added to
        the actual (correct) weight vector. This makes sense, as we want to reject the wrong answer, and accept
        the correct one.

        After the final iteration, the final weight vectors should be somewhat stable (it is of importance to
        note that unlike the assumptions of the binary perceptron, there is no guarantee the multi-class
        perceptron will reach a steady state), and the classifier will be ready to be put to use.
        """
        for _ in xrange(self.iterations):
            for category, feature_dict in self.train_set:
                # Format feature values as a vector, with extra BIAS term.
                feature_vector = np.array(([feature_dict[k] for k in self.feature_list]).append(BIAS))

                # Initialize arg_max value, predicted class.
                arg_max, predicted_class = 0, self.classes[0]

                # Multi-Class Decision Rule:
                for c in self.classes:
                    current_activation = np.dot(feature_vector, self.weight_vectors[c])
                    if current_activation >= arg_max:
                        arg_max, predicted_class = current_activation, c

                # Update Rule:
                if not (category == c):
                    self.weight_vectors[category] += feature_vector
                    self.weight_vectors[c] -= feature_vector

    def save_classifier(self, classifier_name):
        """
        Saves classifier as a .pickle file to the classifier_models directory.

        :param  classifier_name  Name under which to save the classifier.
        """
        with open(OUTPUT_PATH + classifier_name, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_classifier(classifier_name):
        """
        Unpickle the classifier, returns the MultiClassPerceptron object.

        :param  classifier_name  Name the classifier was saved under.
        :return                  Return instance of MultiClassPerceptron.
        """
        with open(OUTPUT_PATH + classifier_name, 'rb') as f:
            return pickle.load(f)


# Simple Sandbox Script to demonstrate entire Pipeline (Loading, Training, Saving, getting Analytics)
if __name__ == "__main__":
    shape_classifier = MultiClassPerceptron(classes, feature_list, feature_data)
    shape_classifier.train()