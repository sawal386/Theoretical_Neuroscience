#Code for Assignment 7. The program computationally analyzes white gaussian noise
import numpy as np 
import matplotlib.pyplot as plt 


def simulate_white_noise(mu, sigma):
	'''simulates a white Gaussian noise process'''

	del_t = 0.1 #delta t in ms
	T = 10 #total simulation time
	t = 0
	time_arr = []
	current_arr = []
	while t < T:
		i = np.random.normal(mu, sigma ** 2, 1)
		current_arr.append(i)
		t += del_t
		time_arr.append(t)

	fig = plt.figure()
	axes = fig.add_subplot(1, 1, 1)
	axes.set_xlabel("Time in ms")
	axes.set_ylabel("Current")
	axes.set_title("Plot of white noise current")
	axes.plot(time_arr, current_arr)
	plt.show()

def plot_autocorrelation(mu, sigma, tau_max):
	'''plots the autocorrelation function of the white Gaussian Noise
	characterized by mean u and variance sigma ** 2'''

	T = 1000
	tau_vals = np.linspace(0, tau_max, 11)
	current_arr = np.random.normal(mu, sigma ** 2, 1000000)
	print(current_arr)
	auto_corr_arr = []
	del_t = 0.1
	for tau in tau_vals:
		step = int(tau / del_t)
		print(step)
		auto_corr = 0
		i = 0
		while i < T / del_t:
			x_t = current_arr[i]
			x_t_tau = current_arr[i + step]
			print(x_t, x_t_tau, tau)
			auto_corr += x_t * x_t_tau
			i += 1
		auto_corr_arr.append(auto_corr / T)

	fig = plt.figure()
	axes = fig.add_subplot(1, 1, 1)
	axes.set_xlabel("$\Delta$" + " in ms")
	axes.set_ylabel("$\mathcal{R}_{N}(t_1, t_2)$")
	axes.set_title("Plot of Autocorrelation function of white Gaussian noise")
	axes.scatter(tau_vals, auto_corr_arr)
	plt.show()

def f_i_values(mu, sigma, H = 0):
	'''returns the f-I curve values'''

	tau_ref = 2
	theta = 20
	t = tau_ref + sigma ** 2 / (2 * mu ** 2) * (np.exp(-2 * mu * theta / (sigma ** 2))
		- np.exp(-2 * mu * H / (sigma ** 2))) + (theta - H) / mu 

	return 1 / t 

def plot_f_I(sigma_values):
	'''plots the f-I curve as a function of mu for different values of sigma'''

	sigma_fi = {}
	mu = np.linspace(-15,15, 1000)
	for s in sigma_values:
		sigma_fi[str(s)] = f_i_values(mu, s)

	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	axes.set_xlabel("$\mu$")
	axes.set_ylabel(r"$\nu$")
	axes.set_title("f-I curve for different values of $\sigma$ plotted as a function of $\mu$")
	for s in sigma_fi:
		axes.plot(mu, sigma_fi[s], label = "$\sigma$ = " +str(s))
	axes.legend()
	plt.show()

def analyze_f_I_H(mu, sigma):
	'''plots the f_I curve for different values of H'''

	tau_ref = 2
	theta = 20
	H_vals = np.linspace(0, theta, 100)
	nu = f_i_values(mu, sigma, H_vals)
	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	axes.set_xlabel("H")
	axes.set_ylabel(r"$\nu$")
	axes.set_title("Relationship between f-I curve and H")
	axes.plot(H_vals, nu)
	plt.show()

if __name__ == '__main__':
	simulate_white_noise(0, 10) #first part of question 1
	plot_autocorrelation(0, 10, 10)
	plot_f_I(np.linspace(1, 10, 8))
	analyze_f_I_H(2, 10)
