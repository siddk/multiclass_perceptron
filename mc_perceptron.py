"""
mc_perceptron.py

A class for building, training, and getting analytics for a Multi-Class Perceptron machine learning model in Python.

Loads feature data and examples from the feature_data sub-directory, in the predefined format (found in the README),
and builds a mc_perceptron model object, to be saved in the classifier_models directory.
"""

__author__ = "Sidd Karamcheti"

# Constants
BIAS = 1                # Dummy Feature for use in setting constant factor in Training.
TRAIN_TEST_RATIO = .75  # Default Ratio of data to be used in Training vs. Testing.
ITERATIONS = 100        # Default Number of Training Iterations.


class MultiClassPerceptron():
    """
    A Multi-Class Perceptron Model object, with functions for loading feature data, training the algorithm,
    and running analytics on model performance.

    :param  classes           List of categories/classes (match tags in tagged data).
    :param  feature_list      List of features.
    :param  feature_data      Feature Data, in format specified in README, usually imported from feature_data.
    :param  train_test_ratio  Ratio of data to be used in training vs. testing. Set to 75% by default.
    :param  iterations        Number of iterations to run training data through. Set to 100 by default.
    """
    def __init__(self, classes, feature_list, feature_data, train_test_ratio=TRAIN_TEST_RATIO, iterations=ITERATIONS):
        pass

