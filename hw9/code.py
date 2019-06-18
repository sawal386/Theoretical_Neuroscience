#The program implements perceptron algorithm and its variants
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin


def compute_dimension(data_mat):
	'''computes the dimension of the data matrix, data_mat. dimension is defined
	   as the number of principal components that explains 95% of the variance'''

	C = np.dot(data_mat,data_mat.T) #covariance matrix
	S, V = lin.eig(C)
	d = 0 #dimension
	var_exp = 0
	while var_exp <= 0.95:
		var_exp += S[d] ** 2 / lin.norm(S) ** 2
		d += 1

	return d

def question_three():
	'''runs question three of the assignment'''

	p = 1000 #total number of data
	N = 50 #dimension of the data
	X = []
	M = 250 #number of neurons in the network
	for i in range(p):
		v = np.random.binomial(1, 0.5, N )
		v = np.where(v == 0, -1, v)
		X.append(v)
	X = np.asarray(X).T
	dim_X = compute_dimension(X)
	print("The dimension is:", dim_X)

    #dimension of the linear map
	W = np.random.normal(0, 1, M * N) #initial weight vector
	W.shape = (M, N)
	y1  = np.dot(W, X)
	dim_y1 = compute_dimension(y1)
	print("The dimension of linear map is:", dim_y1)

    #dimension of rectified linear map without 
	y2 = np.where(y1 <= 0, 0, y1)
	dim_y2 = compute_dimension(y2)
	print("The dimension of rectified linear map is:",dim_y2)

    #theta = 1.80 leads to 10% of the neurons being active
    #measure theta

	y3 = y1 - 1.80
	y3 = np.where(y3 <=0, 0, y3)
	#dimension of rectified linear map with an intercept
	dim_y3 = compute_dimension(y3)
	print("The dimension of rectified linear map is:",dim_y3)


def question_four():
	'''runs question 4 of the assignment'''

	N = 500 #dimension of the matrix J
	J = np.random.randn(N, N)/ N ** 0.5
	eig_vals, eig_vec = lin.eig(J)
	eig_real = eig_vals.real
	eig_imag = eig_vals.imag
	plot_scatter(eig_real, eig_imag, "Re$(\lambda)$", "Im$(\lambda)$", 
		"Plot of Eigenvalues in the complex plane")

	J_1 = J[0,:]
	total_time = 100
	del_t = 0.01
	x_ini = np.random.randn(N)
	G = [1, 1.5, 2, 3]
	x_fin = run_simulation(J, G, total_time, del_t, x_ini)

	fig1 = plt.figure()
	axes1 = fig1.add_subplot(1,1,1)
	axes1.set_xlabel("time, t")
	axes1.set_ylabel("$x_0$")
	axes1.set_title("Neuronal Activity for $x_1$, \n eig_val = " + str(eig_vals[0]))
	
	fig2 = plt.figure()
	axes2 = fig2.add_subplot(1,1,1)	
	axes2.set_xlabel("time, t")
	axes2.set_ylabel("$x_51$")
	axes2.set_title("Neuronal Activity for $x_{51}$,\n eig_val = " + str(eig_vals[50]))
	t_ax = np.linspace(0, x_fin.shape[0] - 1, x_fin.shape[0])

	for i in range(len(G)):
		axes1.plot(t_ax, x_fin[:,0, i], label = "g = " + str(G[i]))
		axes2.plot(t_ax, x_fin[:, 50, i], label = "g = " + str(G[i]))
	axes1.legend()
	axes2.legend()
	plt.show()


def run_simulation(J, g, T, step, x_ini):
	'''simulates the neuronal activity
       J: weight matrix; g: the constant value; T: total time for simulation
       step: time step for Euler method; x_ini: initial values
	'''
	
	stack = []
	for i in range(len(g)):
		stack.append(x_ini)
	stack = tuple(stack)
	activity = []
	t = 0 
	x0 = np.vstack(stack).T
	activity.append(x0)

	while t < T:
		dx_dt = -x0 + g * np.dot(J, np.tanh(x0))
		x1 = x0 + step * dx_dt
		activity.append(x0)
		t += step
		t = round(t, 2)

	return np.asarray(activity)


def plot_scatter(x, y, x_lab, y_lab, title):
	'''plots scatter diagram
	x: values to be plotted in the x-axis
	y: values to be plotted in the y-axis
	x_lab: label of the x-axis
	y: label of the y-axis
	title: title of the plot'''

	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	axes.set_xlabel(x_lab)
	axes.set_ylabel(y_lab)
	axes.scatter(x, y)
	axes.set_title(title)
	plt.show()

def plot_figure(x, y, x_lab, y_lab, title):
	'''plots the desired graph
	x: values to be plotted in the x-axis
	y: values to be plotted in the y-axis
	x_lab: label of the x-axis
	y: label of the y-axis
	title: title of the plot'''

	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	axes.set_xlabel(x_lab)
	axes.set_ylabel(y_lab)
	axes.plot(x, y)
	axes.set_title(title)
	plt.show()

def run_perceptron(X, l, T):
	'''runs perceptron algorithm on a given labelled data set
	X: data matrix; l: labels of the data; T: total time of simulation'''

	dim = X.shape[1]
	n = X.shape[0]
	eta = 0.1 #learning rate
	w = np.zeros((1, dim)) #initialize weight matrix
	theta  = 0
	correct_proportion = [] # # of correctly classified points
	for t in range(T):
		error = compute_error(X, l, w, theta)
		correct_proportion.append(1 - error)
		if error == 0:
			break
		i = np.random.randint(n)
		x = X[[i], :]
		dot = np.dot(w, x.T)[0][0] - theta
		l_hat = compute_sign(dot) # label based on perception algorithm
		if l_hat != l[i]:
			w = w + eta * l[i] * x
			theta = theta - eta * l[i]

	return correct_proportion

		
def compute_error(X, l, w, theta):
	'''computers the error of perceptron algorithm
	X: data matrix; l: labels; w: weight; theta: threshold'''

	incorrect = 0 #number of incorrectly classifed
	whole_dot = np.dot(w, X.T) - theta

	l_hat_whole = compute_sign(whole_dot)
	l_hat_whole.shape = (len(l), )

	l_res = np.abs((l - l_hat_whole)) / 2
	incorrect = np.sum(l_res)

	return incorrect / X.shape[0]

def compute_sign(num):
	'''computes the label(sign) of the data based on perceptron'''

	num = np.where(num >=0, 1, num)
	y = np.where(num < 0, -1, num)

	return y

def initialize_X(num_points, dim):
	'''initialize the data matrix'''

	X = []
	for i in range(num_points):
		x = np.random.binomial(1, 0.5 , dim)
		x = np.where(x == 0, -1, x)
		X.append(x)

	return np.asarray(X)

def part_a():
	'''part a of question 1'''

	N = 100
	p = 10
	labels = np.random.binomial(1, 0.5, p)
	labels = np.where(labels == 0, -1, labels)
	X = initialize_X(p, N)
	total_iter = 100  * p #total number of iterations
	#number of correctly classified points in each interation
	correctly_classified = run_perceptron(X, labels, total_iter)
	iter_ax = np.linspace(1, len(correctly_classified), len(correctly_classified))
	plot_figure(iter_ax, correctly_classified, "iteration #", "Fraction", 
		"Porportion of correctly classified points")

def part_b():
	'''part b of question 1'''

	N_vals = []
	while len(N_vals) != 10:
		n = np.random.randint(100, 300)
		if n not in N_vals:
			N_vals.append(n)

	N_vals = np.sort(N_vals)
	print(N_vals)
	p_max = []

	#choose the appropriate starting value of p
	for N in N_vals:
		print("N:",N)
		p = 100
		if 150 <= N < 250:
			p = 250
		elif 250 <= N < 350:
			p = 400
		elif 350 <= N < 450:
			p = 750
		elif 450 <= N < 550:
			p = 1000

		correct = 1
		while correct > 0.9:
			total_iter = 100 * p
			X = initialize_X(p, N)
			labels = np.random.binomial(1, 0.5, p)
			labels = np.where(labels == 0, -1, labels)
			correctly_classified = run_perceptron(X, labels, total_iter)
			correct_max = np.amax(correctly_classified)
			correct = correct_max
			print(N, p,correct)
			p += 1

		p_max.append(p-2)
		print(p_max)
	#plot_scatter(N_vals, p_max, "N", "p_max", "# of correctly classifid points")
	res = np.polyfit(N_vals, p_max, 1)
	p_max_hat = []
	n_reg = np.linspace(100, 300, 20)
	for N in n_reg:
		p_max_hat.append(N * res[0] + res[1])
	fig1 = plt.figure()
	axes1 = fig1.add_subplot(1,1,1)
	axes1.set_xlabel("N(dimension)")
	axes1.set_ylabel("p_max")
	axes1.set_title("Variation of p_max with respect to N")
	axes1.scatter(N_vals, p_max, label = "p_max")
	axes1.plot(n_reg, p_max_hat, label = "$\hat{p_max_hat}$")
	axes1.legend()
	plt.show()


def question_one():
	'''implements question 1 of the assignment'''

	#part_a()
	part_b()

def question_two():
	'''implements question 2 of the assignment'''

	p = 5
	N = 100
	X = generate_multi_p(N, p)
	label = np.random.binomial(1, 0.5, p **2)
	label = np.where(label == 0, -1, label)
	result = run_perceptron(X, label, 100 * (p **2))

	results_li = []
	N_vals = np.linspace(100, 1000, 91, dtype = int)
	for N1  in N_vals:
		results = run_perceptron(X, label, 100 * (p ** 2))
		average_peformance = np.average(results)
		results_li.append(average_peformance)

	plot_scatter(N_vals, results_li, "N", "Accuracy", 
		"Performance of perceptron on co-related data when varying the dimensionality")
	p_results = []
	p_vals = np.linspace(5, 25, 10, dtype = int)
	p_vals.dtype = int
	print("p_vals:", p_vals)
	for p1 in p_vals:
		X = generate_multi_p(N, p1)
		label = np.random.binomial(1, 0.5, p1 ** 2)
		label = np.where(label == 0, -1, label)
		results = run_perceptron(X, label, 100 * (p1 ** 2))
		avg_per = results[-1]
		print(p1, avg_per)
		p_results.append(avg_per)

	plot_scatter(p_vals, p_results, "p", "Accuracy", 
		"Performance of perceptron on co-related data when varying the data size")



def generate_multi_p(N, P):
	'''generates data points of dim N ** 2 (used for question 2)
	N: dimension
	p: number of data points'''
	
	pattern = []
	for i in range(P):
		x = np.random.binomial(1, 0.5, N)
		x = np.where(x == 0, -1, x)
		pattern.append(x)

	pattern2 = []
	for i in range(len(pattern)):
		for j in range(len(pattern)):
			combined = np.hstack((pattern[i], pattern[j]))
			pattern2.append(combined)
	print(pattern2[0].shape)

	return np.asarray(pattern2)

if __name__ == "__main__":
	#question_four()
	question_three()
	#question_one()
	#question_two()
