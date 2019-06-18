#This program implements stochastic gradient descent and noisy stochastic
#gradient descent

import numpy as np 
import matplotlib.pyplot as plt

def question_1():

	x = np.linspace(-3, 3, 1000)
	y = f(x)
	grad_des_x = run_gradient_descent(3, 3000, 0.001)
	
	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	axes.set_xlabel("x")
	axes.set_ylabel("f(x)")
	axes.set_title("Question 1 plots")
	axes.plot(x, y, label = "(x, f(x))")
	axes.plot(grad_des_x, f(grad_des_x), label = "$(x_n, f(x_n))$", color = "red")
	size_n = [20,25,30]
	fig2 = plt.figure()
	axes2 = fig2.add_subplot(1,1,1)
	axes2.set_xlabel("x")
	axes2.set_ylabel("f(x)")
	axes2.set_title("Gradient Descent with noise")
	axes2.plot(x, y, label = "(x, f(x)", color = "red")
	
	noise_grad_des_x = run_noisy_gradient_descent(3, 0.2, 3000, 0.001)
	axes2.scatter(noise_grad_des_x, f(noise_grad_des_x), 
		label = "$(x_n, f(x_n))$ with noise $\\xi $ #")
	axes2.scatter(noise_grad_des_x[-1], f(noise_grad_des_x[-1]), 
		color = "black", label = "final point")
	
	axes.legend()
	axes2.legend()
	plt.show()

def f(x):

	return x ** 4 - 8 * (x ** 2) + 3 * x + 16

def df_dx(x):

	return 4 * (x ** 3) - 16 * x + 3

def run_gradient_descent(x_0, N, eta):

	X = [x_0]
	for i in range(N):
		x_1 = x_0 - eta * df_dx(x_0)
		x_0 = x_1
		X.append(x_0)

	return np.asarray(X)

def run_noisy_gradient_descent(x_0, sigma_0, N, eta):
	X = [x_0]

	for i in range(N):
		n =  np.random.normal(0, sigma_0)
		x_1 = x_0 - eta * df_dx(x_0) + n
		print(n, sigma_0)
		x_0 = x_1
		sigma_0 = 0.999 * sigma_0
		X.append(x_0)

	return np.asarray(X)

def question_2():

	label1_x = [-1, 1, -2]
	label2_x = [2, 4, 1]
	label1_y = [-2, 3, 0]
	label2_y = [-3, 2, -2]

	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	axes.set_xlabel("x")
	axes.set_ylabel("y")
	axes.set_title("Question 2 plot")
	axes.scatter(label1_x, label1_y, color = "red", label = "l = 1")
	axes.scatter(label2_x, label2_y, color = "blue", label = "l = -1")
	axes.text(1, -2, "SV")
	axes.text(-1, -2, "SV")
	plt.show()

if __name__ == "__main__":
	question_1()
	#question_2()
