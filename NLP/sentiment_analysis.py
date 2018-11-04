'''
WORK - IN - PROGRESS
Dataset: IMDB sentiment classification dataset
Each training sample consists of a sequence of integers, each of 
which represents a word. The integer represents the frequency of the
word in the database
Each label a binary class, representing whether the review was 
postive or negative
'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb

def vectorize(sequences, dimension = 10000):
	result = np.zeros((len(sequences), dimension))
	for i, seq in enumerate(sequences):
		result[i, seq] = 1
	return result


(x_train, y_train), (x_test, y_test) = \
imdb.load_data(num_words=10000) # Consider the top 10,000 most common words

X = np.concatenate((x_train, x_test), axis=0)
Y = np.concatenate((y_train, y_test), axis=0)

# Understanding the data
print 'Categories:', np.unique(Y)
print 'No of unique words:', len(np.unique(np.hstack(X)))
print 'Most frequently occuring word:', max(np.unique(np.hstack(X)))

# Converting all input data to same number of words for the NN

X = vectorize(X)
