import numpy as np

def nonlin(x, deriv=False):
	if deriv:
		return x*(1 - x)
	return 1./(1 + np.exp(-x))

# data
X = np.array(
	[[0, 0, 1],
	[0, 1, 1],
	[1, 0, 1],
	[1, 1, 1]
	])
y = np.array([[0, 1, 1, 0]]).T

# initialize weights
w0 = np.random.random((3, 25))
w1 = np.random.random((25, 1))

for i in xrange(100000):
	# feed forward
	l0 = X
	l1 = nonlin(np.dot(l0, w0))
	l2 = nonlin(np.dot(l1, w1))

	# error
	l2_error = y - l2
	l2_delta = l2_error * nonlin(l2, deriv=True)

	l1_error = l2_delta.dot(w1.T)
	l1_delta = l1_error * nonlin(l1, deriv=True)

	w1+= np.dot(l1.T, l2_delta)
	w0+= np.dot(l0.T, l1_delta)
	
	if i%10000==0:
		print "Error:", np.mean(np.abs(l2_error))