'''
Description: Training a CNN from scratch.
Dataset: MNist
'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

print "---------------"

batch_size = 128
epochs = 12

(x_train, y_train), (x_test, y_test) = mnist.load_data()

n_train, rows, cols = np.shape(x_train)
n_test, _, _ = np.shape(x_test)

# Reshaping training data to n_samples*(rows, cols, ch)
x_train = x_train.reshape(n_train, rows, cols, 1)
x_test = x_test.reshape(n_test, rows, cols, 1)

print x_train.shape
print x_test.shape

# Converting labels to a binary representation
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Creating the model
model = Sequential()

# Add a 2D convolutional layer
model.add(Conv2D(filters=32,
				 kernel_size=(3,3),
				 activation= 'relu',
				 input_shape=(rows,cols,1)))

# Add another 2D convolutional layer
model.add(Conv2D(filters=64,
				 kernel_size=(3,3),
				 activation='relu'))

# Add a max pooling layer which finds max element in (2,2) windows
model.add(MaxPooling2D(pool_size=(2,2)))

# Switch-off 25% of neurons to generalize, prevent overfitting
model.add(Dropout(rate=0.25))

# Creates a vector of its input
model.add(Flatten())

# Densely connected layer, output=activation(dot(input, weights)+bias)
model.add(Dense(units=128,
				activation='relu'))

# Switch-off 50% of neurons
model.add(Dropout(rate=0.5))

# Densely connected layer, mapping to class labels
model.add(Dense(units=num_classes,
				activation='softmax'))

# Compile with an Adadelta optimizer (adapts learning rates based
# on a window of gradients)
model.compile(loss=categorical_crossentropy,
			  optimizer=optimizers.Adadelta(),
			  metrics=['accuracy'])

history = model.fit(x_train, y_train,
		  			batch_size=batch_size,
		  			epochs=epochs,
		  			verbose=1,
		  			validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print 'Test loss:', score[0]
print 'Test accuracy:', score[1]

print history.history.keys()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print history.history.keys()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()