'''
Ref: http://neuralnetworksanddeeplearning.com
'''
import random
import numpy as np
import mnist_loader

class Network(object):
	def __init__(self, sizes):
		'''
		sizes: #neurons in the respective layers
		If sizes is [2,3,1], biases:[3,1], weights:[[3*2], [1*3]]
		Weights and biases are initialized randomly usign a Gaussian distribution
		First layer is the input layer, so no weights or biases are set for those neurons
		'''
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]] # Every neuron has a bias
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])] # Size is #neurons in cuurent*next layer


	def SGD(self, training_data, epochs, mini_batch_size, eta,
		test_data = None):
		'''
		Trains the network using mini-batch stochastic gradient descent
		'training_data': is a list of tuples '(x,y)' representing the training 
		input and its output.
		'test_data': If provided, the network will be evaluated after every
		epoch, and progress will be printed.
		1. Loop over the #epochs
		2. Divide data into batches
		3. Loop over #batches
		4. Perform SGD: Loop over sample, backpropogate error, update weights/biases
		'''
		if test_data: n_test = len(test_data)
		n = len(training_data)
		for j in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size]
			for k in xrange(0,n,mini_batch_size)]
			
			for mini_batch in mini_batches: # Run the SGD
				self.update_mini_batch(mini_batch, eta)
	
			if test_data: # If performance is to be evaluated after every epoch
				print 'Epoch {0}: {1} / {2}'.format(
					j, self.evaluate(test_data), n_test)
			else:
				print 'Epoch {0} complete'.format(j)


	def update_mini_batch(self, mini_batch, eta):
		'''
		Updates the network's weights and biases by applying gradient descent
		using backpropogation to a single batch
		mini_batch: list of tuples (x,y)
		eta: learning rate
		1. Initialise nabla w,b to 0s
		2. Loop over each training data/label
		3. Backpropogate error: Calculate 'a', 'delta', 'delta_nabla_w/b'
		   We have the nabla_b/w for each sample
		4. Update the weights and biases for this batch
		'''
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x,y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x,y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w - (eta/len(mini_batch)) * nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b - (eta/len(mini_batch)) * nb
						for b, nb in zip(self.biases, nabla_b)]	

	def backprop(self, x, y):
		'''
		x: Training data; y: Training label
		1. Forward pass: Calculate activation
		2. Output error: Calculate delta for the last layer
		3. Backward pass: Calculate delta for each layer,
						  Calculate the nabla w and b for each layers
		'''
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# Forward-pass
		activation = x # Initially, this is the training data
		activations = [x]
		zs = []
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w,activation) + b
			zs.append(z)
			activation = self.sigmoid(z) # Activation for the current layer is this
			activations.append(activation)

		# Output error
		delta = self.cost_derivative(activations[-1],y) * self.sigmoid_prime(zs[-1])

		# Backward-pass
		  # Last layer values
		nabla_b[-1] = delta 
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		for l in xrange(2, self.num_layers):
			z = zs[-l]
			delta = np.dot(self.weights[-l+1].transpose(), delta) * self.sigmoid_prime(z)
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def cost_derivative(self, output_activation, y):
		return (output_activation - y)


	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))


	def sigmoid_prime(self, z):
		return self.sigmoid(z) * (1-self.sigmoid(z))


	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
		return sum(int(x==y) for (x,y) in test_results)


	def feedforward(self,a):
		'''
		For every pair of weights and biases between layers, 
		an activation array is present.
		Returns the output of the network  if 'a' is input
		'''
		for b,w in zip(self.biases, self.weights):
			a = self.sigmoid(np.dot(w,a) + b)
		return a
		