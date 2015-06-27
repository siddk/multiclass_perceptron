# Multi-Class Perceptron

The multi-class perceptron algorithm is a supervised learning algorithm for classification of data into one of a series
of classes. The following implementation was built as part of my project to build a domain-specific natural language
question answering algorithm (interview_qa), to classify questions into categories based on their content.

This algorithm is built in such a way that it can be generalized to any use-case, with details on how to format data
in the sections below.

### Algorithm Summary ###

This algorithm, like most perceptron algorithms is based on the biological model of a neuron, and it's activation. In
the case of a normal perceptron (binary classifier), the data is broken up into a series of attributes, or features,
each with a specific value. When this feature vector is received by the artificial neuron as a stimulus, it is
multiplied (dot product) by a weight vector, to calculate the activation value of the specific data point. If the
activation energy is high enough, the neuron fires (the data meets the classification criteria).

In the case of a multi-class perceptron, things are a little different. The data comes in the same way, but instead of
the respecting feature vector being multiplied by a single weight vector (for a single class), it is multiplied
(dot product) by a number of weight vectors (a separate vector of weights for each unique class). Whichever weight vector
that yields the highest activation energy product is the class the data belongs to.