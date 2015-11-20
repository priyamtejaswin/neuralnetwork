import numpy as np
import sys
from pprint import pprint
from sklearn import datasets

def onehot(x, n=10):
	"""Returns target values in onehot format."""
	z = np.zeros((x.shape[0], n))
	for row,val in zip(z,x):
		row[val] = 1.
	return z

def softmax(x):
	"""Returns softmax probability distribution."""
	## Not working correctly - the error calculation needs to be changed - come back later.
	e_x = np.exp(x)
	out = e_x / np.sum(e_x, axis=0)
	return out

def tanh(x):
	"""Tanh activation."""
	return np.tanh(x)

def tanh_deriv(x):
	"""Tanh activation derivative."""
	return (1.0 - np.tanh(x)**2)

def sigmoid(x):
	"""Sigmoid activation."""
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_deriv(x):
	"""Sigmoid derivative."""
	return sigmoid(x) * (1 - sigmoid(x))

class CrossEntropyCost(object):
	"""Creates a CrossEntropy cost object.
	Contains a object.value and object.deriv function."""
	def value(self, a, y):
		return np.mean(np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)), axis=0))

	def deriv(self, z, a, y, nonlin_deriv):
		return (a-y)

class QuadraticCost(object):
	"""Creates a quadratic cost object.
	Contains a object.value function and object.deriv function."""
	def value(self, a, y):
		"""QuadraticCost cost value."""
		return np.mean(0.5*np.sum((a-y)**2, axis=0))

	def deriv(self, z, a, y, nonlin_deriv):
		"""QuadraticCost cost derivative."""
		return (a-y) * nonlin_deriv(z)

class NeuralNetwork(object):
	"""Creates a artificial neural network object."""
	def __init__(self, layers, batch_size=2, cost=QuadraticCost(), nonlin="sigmoid", use_softmax=False):
		if nonlin=="sigmoid":
			self.nonlin = sigmoid
			self.nonlin_deriv = sigmoid_deriv
		elif nonlin=="tanh":
			self.nonlin = tanh
			self.nonlin_deriv = tanh_deriv
		self.use_softmax = use_softmax
		self.cost = cost
		self.batch_size = batch_size
		self.layers = layers
		self.weights = [np.random.randn(j, i) for i,j in zip(layers[:-1], layers[1:])]
		self.biases = [np.random.randn(j, batch_size) for j in layers[1:]]

	def properties(self):
		print "\n--network properties--\n"
		table = [
				["number of layers:", len(self.layers)],
				["input units:", self.layers[0]],
				["output units:", self.layers[-1]],
				["number of hidden units", self.layers[1:-1]],
				["batch size:", self.batch_size],
				["softmax output:", self.use_softmax],
				["cost:", self.cost],
				["non-linearity:", self.nonlin]
				]
		pprint(table)

	def feedforward(self, a):
		"""Feed forward output - also used in accuracy evaluation."""
		activations, weighted_inputs = [a], []
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, a) + b
			a = self.nonlin(z)
			activations.append(a)
			weighted_inputs.append(a)
		if self.use_softmax:
			activations[-1] = softmax(weighted_inputs[-1])
		return activations, weighted_inputs

	def backprop(self, output_activation, weighted_inputs, target):
		"""Backpropagates the error from the output layer to all hidden layers."""
		del_out = self.cost.deriv(weighted_inputs[-1], output_activation, target, self.nonlin_deriv)
		del_array = [del_out]
		for w, z in zip(self.weights[::-1], weighted_inputs[-2::-1]):
			del_array.append(np.dot(w.T, del_array[-1]) * self.nonlin_deriv(z))
		return del_array

	def update_params(self, del_array, activations):
		"""Udates the model parameters(w,b)."""
		for w,b,del_val,a in zip(self.weights[::-1], self.biases[::-1], del_array, activations[-2::-1]):
			w -= self.alpha * np.dot(del_val, a.T)
			b -= self.alpha * del_val

	def sgd(self, trX, trY, teX=None, teY=None, epochs=500, alpha=0.05):
		"""Performs stochastic gradient descent for a given bactch size.
		This method also takes a cost object for calculating cost and cost derivative."""
		self.epochs, self.alpha = epochs, alpha
		mi_trX = trX.shape[0] - trX.shape[0]%self.batch_size
		start = xrange(0, mi_trX, self.batch_size)
		stop = xrange(self.batch_size, mi_trX+1, self.batch_size)

		print "\ntraining...\n"
		for _i_ in xrange(epochs+1):
			for i,j in zip(start, stop):
				activations, weighted_inputs = self.feedforward(trX[i:j, :].T)
				del_array = self.backprop(activations[-1], weighted_inputs, trY[:, i:j])
				self.update_params(del_array, activations)

			if _i_%50==0:
				print "epoch:", _i_, "error:", self.cost.value(activations[-1], trY[:, i:j])
				self.evaluate(teX, teY)

	def evaluate(self, teX, teY):
		"""Uses the feedforward method to calculate model accuracy on testing data."""
		mi_teX = teX.shape[0] - teX.shape[0]%self.batch_size
		means_array = []
		start = xrange(0, mi_teX, self.batch_size)
		stop = xrange(self.batch_size, mi_teX+1, self.batch_size)
		for i,j in zip(start, stop):
			activations, _ = self.feedforward(teX[i:j, :].T)
			means_array.append(np.mean(np.argmax(activations[-1], axis=0)==np.argmax(teY[:, i:j], axis=0)))
		print "accuracy:", np.mean(means_array), "\n---"

## ann code ends here - below are helper functions for testing the model on datasets

def test_iris():
	iris = datasets.load_iris()
	print "iris data shape:", iris.data.shape
	print "iris target shape:", iris.target.shape

	data = np.concatenate((iris.data, onehot(iris.target, 3)), axis=1)
	np.random.shuffle(data)

	X,Y = data[:, :4]/np.max(data), data[:, 4:]
	trX, trY = X[:100, :], Y[:100, :].T
	teX, teY = X[100:, :], Y[100:, :].T
	print "train data:", trX.shape, trY.shape
	print "test data:", teX.shape, teY.shape

	nn = NeuralNetwork([4, 50, 50, 3], batch_size=10)
	nn.properties()
	nn.sgd(trX, trY, teX, teY, epochs=1500, alpha=0.05)

def test_digits():
	digits = datasets.load_digits()
	print "digits data shape:", digits.data.shape
	print "digits target shape:", digits.target.shape

	data = np.concatenate((digits.data, onehot(digits.target, 10)), axis=1)
	np.random.shuffle(data)

	X,Y = data[:, :64]/np.max(data), data[:, 64:]
	trX, trY = X[:1500, :], Y[:1500, :].T
	teX, teY = X[1500:, :], Y[1500:, :].T
	print "train data:", trX.shape, trY.shape
	print "test data:", teX.shape, teY.shape

	nn = NeuralNetwork([64, 100, 100, 100, 10], batch_size=1, nonlin="tanh", cost=CrossEntropyCost(), use_softmax=False)
	nn.properties()
	nn.sgd(trX, trY, teX, teY, epochs=1000, alpha=0.025)

if __name__ == '__main__':
	# training data
	# X = np.array(
	# 	[[0, 0, 1],
	# 	[0, 1, 1],
	# 	[1, 0, 1],
	# 	[1, 1, 1]
	# 	])
	# y = np.array([[0, 1, 1, 0]])
	test_digits()