'''
Ref: https://medium.com/@connectwithghosh/simple-autoencoder-example-using-tensorflow-in-python-on-the-fashion-mnist-dataset-eee63b8ed9f1
Data: https://www.kaggle.com/zalando-research/fashionmnist
Objective: Autoencoders for dimensionality reduction
'''

import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


train = np.loadtxt('fashion-mnist_train.csv', delimiter=',', skiprows=1)[:,1:]
# train = train.drop(['label'], axis = 1)
print 'Data:', train.shape

# Defining number of nodes
n_nodes_ip = 784 # Each image has 784 pixels
n_nodes_h11 = 32 # Encoder has 1 hidden layer 
n_nodes_h12 = 32 # Decoder has 1 hidden layer
n_nodes_op = 784 # Output image also has 784 pixels

# Initializing weights and biases
hidden1_val = {'weights': tf.Variable(tf.random_normal([n_nodes_ip, n_nodes_h11])),
		   'biases': tf.Variable(tf.random_normal([n_nodes_h11]))}
hidden2_val = {'weights': tf.Variable(tf.random_normal([n_nodes_h11, n_nodes_h12])),
		   'biases': tf.Variable(tf.random_normal([n_nodes_h12]))}
output_val = {'weights': tf.Variable(tf.random_normal([n_nodes_h12, n_nodes_op])),
		   'biases': tf.Variable(tf.random_normal([n_nodes_op]))}

# Defining the neural net
input_layer = tf.placeholder('float', [None, n_nodes_ip])

layer_1 = tf.nn.sigmoid(
			tf.add(
					tf.matmul(input_layer, hidden1_val['weights']),
					hidden1_val['biases']
				)
			) # wx + b
layer_2 = tf.nn.sigmoid(
			tf.add(
					tf.matmul(layer_1, hidden2_val['weights']),
					hidden2_val['biases']
				)
			)

output_layer = tf.add(
				tf.matmul(layer_2, output_val['weights']),
				output_val['biases']
				)
# Ground truth (input image) for error calc
output_true = tf.placeholder('float', [None, 784])

# Cost function
meansq = tf.reduce_mean(tf.square(output_layer - output_true))

# Optimizer parameters
learn_rate = 0.1
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)

# Running a session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
n_epochs = 10
n_images = train.shape[0]

# Running the model
for epoch in range(n_epochs):
	epoch_loss = 0 # Initializing error as 0
	for i in range(int(n_images / batch_size)): # Running for 1 batch at a time
		epoch_x = train[i*batch_size : (i+1)*batch_size] # Extracting 1 abtch at a time
		_, c = sess.run([optimizer, meansq], 
						feed_dict={input_layer:epoch_x, output_true:epoch_x})
		epoch_loss += c

# Test an image
img_ip = train[999]
	# Running it through the autoencoder
img_op = sess.run(output_layer, feed_dict={input_layer:[img_ip]}) 
	# Running it through just the encoder
encoded_img = sess.run(layer_1, feed_dict={input_layer:[img_ip]})
plt.subplot(121)
plt.imshow(img_ip.reshape(28, 28), cmap='Greys')
plt.subplot(122)
plt.imshow(img_op.reshape(28, 28), cmap='Greys')
plt.show()
