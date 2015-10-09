import numpy as np

class NeuralNetwork(object):
	def __init__(self, layers, batch_size=2):
		self.batch_size = batch_size
		self.layers = layers
		self.weights = [np.random.randn(j, i) for i,j in zip(layers[:-1], layers[1:])]
		self.biases = [np.random.randn(j, batch_size) for j in layers[1:]]

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

	def sgd(self, training_samples, target, epochs=50001, alpha=0.05):
		print "\ntraining...\n"
		for _i_ in xrange(epochs):
			_m = training_samples.shape[0] - training_samples.shape[0]%self.batch_size
			for start, stop in zip(range(0,_m,self.batch_size), range(self.batch_size,_m+1,self.batch_size)):
				activations, weighted_inputs = self.feedforward(training_samples[start:stop, :].T)
				del_array = self.backprop(activations[-1], weighted_inputs, target[:, start:stop])
				for w,b,del_val,a in zip(self.weights[::-1], self.biases[::-1], del_array, activations[-2::-1]):
					w -= alpha * np.dot(del_val, a.T)
					b -= alpha * del_val

			if _i_%10000==0:
				print "epoch:", _i_, "mse:", np.mean(np.square(activations[-1] - target[:, start:stop]))

		print "\nnetwork output"
		print activations[-1]

if __name__ == '__main__':
	# training data
	X = np.array(
		[[0, 0, 1],
		[0, 1, 1],
		[1, 0, 1],
		[1, 1, 1]
		])
	y = np.array([[0, 1, 1, 0]])

	nn = NeuralNetwork([3, 10, 10, 1])
	nn.sgd(X, y)