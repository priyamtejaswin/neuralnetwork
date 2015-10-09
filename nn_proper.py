import numpy as np

def cost(output_activation, y):
	return 0.5 * (y - output_activation)**2

def cost_deriv(output_activation, y):
	return (output_activation - y)

def sigmoid(z):
	return np.array(1./(1. + np.exp(-z)))

def sigmoid_deriv(x):
	return sigmoid(x) * (1 - sigmoid(x))

# training data
X = np.array(
	[[0, 0, 1],
	[0, 1, 1],
	[1, 0, 1],
	[1, 1, 1]
	])
y = np.array([[0, 1, 1, 0]])

# (some)hyperparameters
alpha = 0.05
hidden_units = 5
input_units = 3
output_units = 1
batch_size = 4
# three layer NN
# input, hidden, output
# --1--, --2---, --3---
w2, b2 = np.random.randn(hidden_units, input_units), np.random.randn(hidden_units, batch_size)
w3, b3 = np.random.randn(output_units, hidden_units), np.random.randn(output_units, batch_size)

print "\ntraining...\n"
for epoch in xrange(50001):
	a1 = X.T # activation of first layer
	z2 = np.dot(w2, a1) + b2 # weighted input to second layer
	a2 = sigmoid(z2) # activation of second layer
	z3 = np.dot(w3, a2) + b3 # weighted input to third layer
	a3 = sigmoid(z3) # activation of the third layer

	# backprop
	del3 = cost_deriv(a3, y) * sigmoid_deriv(z3)
	del2 = np.dot(w3.T, del3) * sigmoid_deriv(z2)
#   del1 = np.dot(w2.T, del2) * sigmoid_deriv(z1)
	# gradient descent
	w3 -= (alpha) * np.dot(del3, a2.T)
	b3 -= (alpha) * del3
	w2 -= (alpha) * np.dot(del2, a1.T)
	b2 -= (alpha) * del2

	# error check
	print "del shapes", [x.shape for x in [del3,del2]]
	print w3.shape, b3.shape, del3.shape, a2.shape
	print w2.shape, b2.shape, del2.shape, a1.shape
	raw_input('---')

	if epoch%10000==0:
		print "epoch:", epoch
		print "mean square error:", np.abs(np.mean(np.square(a3 - y)))

print "\n--NETWORK OUTPUT--\n"
print a3, '\n', y