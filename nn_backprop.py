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
y = np.array([0, 1, 1, 0])
print "--INPUT--\t--OUTPUT--"
for _in, _out in zip(X, y):
	print _in, '\t', _out	

# (some)hyperparameters
alpha = 0.05
h_units = 50
# three layer NN
# input, hidden, output
# --1--, --2---, --3---
w2, b2 = np.random.randn(h_units, 3), np.random.randn(h_units, 1)
w3, b3 = np.random.randn(1, h_units), np.random.randn(1, 1)

print "\ntraining...\n"
for epoch in xrange(100001):
	sample = np.random.randint(0, 4)
	a1 = np.matrix(X[sample]).T # activation of first layer
	z2 = np.dot(w2, a1) + b2 # weighted input to second layer
	a2 = sigmoid(z2) # activation of second layer
	z3 = np.dot(w3, a2) + b3 # weighted input to third layer
	a3 = sigmoid(z3) # activation of the third layer

	# backprop
	del3 = cost_deriv(a3, y[sample]) * sigmoid_deriv(z3)
	del2 = np.dot(w3.T, del3) * sigmoid_deriv(z2)

	# gradient descent
	w3 -= (alpha) * np.dot(del3, a2.T)
	b3 -= (alpha) * del3
	w2 -= (alpha) * np.dot(del2, a1.T)
	b2 -= (alpha) * del2

	# error check
	if epoch%10000==0:
		print "epoch:", epoch
		temp_error, temp_out = [], []
		for _in, _out in zip(X, y):
			e_h = sigmoid(np.dot(w2, np.matrix(_in).T) + b2)
			e_o = sigmoid(np.dot(w3, e_h) + b3)
			temp_error.append(e_o - _out)
			temp_out.append(e_o)
		print "mean square error:", np.mean(np.square(temp_error))

print "\n--NETWORK OUTPUT--"
for _e,_out in zip(np.array(temp_out).tolist(), y):
	print _e, _out