# Multi-Class Perceptron

The multi-class perceptron algorithm is a supervised learning algorithm for classification of data into one of a series
of classes. The following implementation was built as part of my project to build a domain-specific natural language
question answering algorithm (interview_qa), to classify questions into categories based on their content.

This algorithm is built in such a way that it can be generalized to any use-case, with details on how to format data
in the sections below. It is meant to be easy to use and understand, without any significant performance issues.

For added benefit, this module also contains functions to facilitate training, building, and testing the classifier,
providing useful metrics and statistics to judge performance.

## Algorithm Summary ##

This algorithm, like most perceptron algorithms is based on the biological model of a neuron, and it's activation. In
the case of a normal perceptron (binary classifier), the data is broken up into a series of attributes, or features,
each with a specific value. When this feature vector is received by the artificial neuron as a stimulus, it is
multiplied (dot product) by a weight vector, to calculate the activation value of the specific data point. If the
activation energy is high enough, the neuron fires (the data meets the classification criteria).

In the case of a multi-class perceptron, things are a little different. The data comes in the same way, but instead of
the respecting feature vector being multiplied by a single weight vector (for a single class), it is multiplied
(dot product) by a number of weight vectors (a separate vector of weights for each unique class). Whichever weight vector
that yields the highest activation energy product is the class the data belongs to. This decision process is known as
the Multi-Class Decision Rule.

### Training Process ###

To train the algorithm, the following process is taken. Unlike some other popular classification algorithms that require
a single pass through the supervised data set (like Naive Bayes), the multi-class perceptron algorithm requires multiple
training iterations to fully learn the data. The iteration count can be easily set as a parameter.

During each iteration of training, the data (formatted as a feature vector) is read in, and the dot product is taken
with each unique weight vector (which are all initially set to 0). The class that yields the highest product is the class
to which the data belongs. In the case this class is the correct value (matches with the actual category to which the
data belongs), nothing happens, and the next data point is read in. However, in the case that the predicted value is
wrong, the weight vectors are corrected as follows: The feature vector is subtracted from the predicted weight vector,
and added to the actual (correct) weight vector. This makes sense, as we want to reject the wrong answer, and accept the
correct one.

After the final iteration, the final weight vectors should be somewhat stable (it is of importance to note that unlike
the assumptions of the binary perceptron, there is no guarantee the multi-class perceptron will reach a steady state),
and the classifier will be ready to be put to use.

------------------------------------------------------------------------------------------------------------------------

## Building a Classifier ##

The following sections detail how to format the data for use with the classifier builder, as well as how to train and
save the classifier for later use. The last section deals with how to build an analytics report for the data.