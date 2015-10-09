import numpy as np
import sys
from pprint import pprint

def onehot(x, n=10):
	z = np.zeros((x.shape[0], n))
	for row,val in zip(z,x):
		row[val] = 1.
	return z

class NeuralNetwork(object):
	def __init__(self, layers, batch_size=2):
		self.batch_size = batch_size
		self.layers = layers
		self.weights = [np.random.randn(j, i) for i,j in zip(layers[:-1], layers[1:])]
		self.biases = [np.random.randn(j, batch_size) for j in layers[1:]]

	def properties(self):
		print "\n--network properties--\n"
		table = [
				["number of layers", len(self.layers)],
				["input units", self.layers[0]],
				["output units", self.layers[-1]],
				["number of hidden units", self.layers[1:-1]]
				]
		pprint(table)

	def cost_deriv(self, output_activation, y):
		return (output_activation - y)

	def sigmoid(self, x):
		return np.array(1./(1. + np.exp(-x)))

	def sigmoid_deriv(self, x):
		return self.sigmoid(x) * (1 - self.sigmoid(x))

	def feedforward(self, a):
		activations, weighted_inputs = [a], []
		for w, b in zip(self.weights, self.biases):
			z = np.dot(w, a) + b
			a = self.sigmoid(z)
			activations.append(a)
			weighted_inputs.append(a)
		return activations, weighted_inputs

	def backprop(self, output_activation, weighted_inputs, target):
		del_out = self.cost_deriv(output_activation, target) * self.sigmoid_deriv(weighted_inputs[-1])
		del_array = [del_out]
		for w, z in zip(self.weights[::-1], weighted_inputs[-2::-1]):
			del_array.append(np.dot(w.T, del_array[-1]) * self.sigmoid_deriv(z))
		return del_array

	def sgd(self, trX, trY, teX=None, teY=None, epochs=500, alpha=0.05):
		print "\ntraining...\n"
		mi_trX = trX.shape[0] - trX.shape[0]%self.batch_size
		for _i_ in xrange(epochs+1):
			for i,j in zip(range(0, mi_trX, self.batch_size), range(self.batch_size, mi_trX+1, self.batch_size)):
				activations, weighted_inputs = self.feedforward(trX[i:j, :].T)
				del_array = self.backprop(activations[-1], weighted_inputs, trY[:, i:j])
				for w,b,del_val,a in zip(self.weights[::-1], self.biases[::-1], del_array, activations[-2::-1]):
					w -= alpha * np.dot(del_val, a.T)
					b -= alpha * del_val

			if _i_%100==0:
				if teX.all():
					print "epoch:", _i_
					self.evaluate(teX, teY)
				else:
					print "\nepoch:", _i_
					print "network\ttarget"
					for col1,col2 in zip(activations[-1].T, trY[:, i:j].T):
						print np.argmax(col1) , "\t", np.argmax(col2)
					raw_input('---')

	def evaluate(self, teX, teY):
		mi_teX = teX.shape[0] - teX.shape[0]%self.batch_size
		means_array = []
		for i,j in zip(range(0, mi_teX, self.batch_size), range(self.batch_size, mi_teX+1, self.batch_size)):
			activations, _ = self.feedforward(teX[i:j, :].T)
			means_array.append(np.mean(np.argmax(activations[-1], axis=0)==np.argmax(teY[:, i:j], axis=0)))
		print "accuracy:", np.mean(means_array), "\n---"

if __name__ == '__main__':
	# training data
	# X = np.array(
	# 	[[0, 0, 1],
	# 	[0, 1, 1],
	# 	[1, 0, 1],
	# 	[1, 1, 1]
	# 	])
	# y = np.array([[0, 1, 1, 0]])

	from sklearn import datasets
	iris = datasets.load_iris()
	print "iris data shape:", iris.data.shape
	print "iris target shape:", iris.target.shape

	data = np.concatenate((iris.data, onehot(iris.target, 3)), axis=1)
	np.random.shuffle(data)
	# sys.exit()
	X,Y = data[:, :4]/np.max(data), data[:, 4:]
	trX, trY = X[:100, :], Y[:100, :].T
	teX, teY = X[100:, :], Y[100:, :].T
	print "train data:", trX.shape, trY.shape
	print "test data:", teX.shape, teY.shape

	nn = NeuralNetwork([4, 50, 50, 3], batch_size=10)
	nn.properties()
	nn.sgd(trX, trY, teX, teY, epochs=1000)