#This program simulates the pair of ODE's to find the value of k at which the
# oscillatory behavior stops

import numpy
import matplotlib.pyplot as plt 

def f(x, y):
	'''corresponds to dx/dt; x and y are positions and t is time'''

	return - x * (x - 1) * (x + 1) + y

def g(x, y, k):
	'''corresponds to dy/dt; x and y are positions and k is a constant'''

	return - k * (2 * x + y) 

def simulate_ode(x_ini, y_ini, del_t, T):
	''' simulates an ODE using Euler integration
	x_ini, y_ini initial values of x and y
	del_t: time step, T: total time for which the simulation is run'''

	k_vals = numpy.linspace(0.01,1.99,199)
	k_not_oscillating = [] #values of k for which the system does not oscillate
	epsilon = 1e-7
	for k in k_vals:
	    t = 0 #the time variable
	    x_0 = x_ini #initial value of x
	    y_0 = y_ini #initial value of y
	    
	    while t <= T:
	    	x_1 = x_0 + f(x_0, y_0) * del_t
	    	y_1 = y_0 + g(x_0, y_0, k) * del_t

	    	#implies f(x, y) and g(x, y) = 0 i.e. x_0 = x_1 and y_0 = y_1
	    	#this means the system is not oscillating. if oscillating, the
	    	# x and y values are changing continuously with respect to time. 
	    	if numpy.abs(x_1 - x_0) <= epsilon and numpy.abs(y_1 - y_0) <= epsilon:
	    		print("equal")
	    		k_not_oscillating.append(k)
	    		break
	    	x_0 = x_1
	    	y_0 = y_1
	    	x_li.append(x_0)
	    	y_li.append(y_0)
	    	t_li.append(t)
	    	t += del_t

	    #print(k)
	#the first value of k for which the system transitions from stable to unstable
	print(k_not_oscillating[0])

simulate_ode(2,1,0.01, 2500)







