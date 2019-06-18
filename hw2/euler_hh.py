#simulates Hodgkin Huxley model using Euler integration
#name: Sawal Acharya  uni: sa3330

import numpy as np 
import matplotlib.pyplot as plt

def alpha_n(V):
    '''returns the alpha value associated with gating variable n'''

    return 0.01 * (V + 55.0) / (1.0 - np.exp(-0.1 * (V + 55)))

def beta_n(V):
    '''returns the beta value associated with gating variable n'''

    return 0.125 * np.exp(-0.0125 * ( V + 65))

def alpha_m(V):
    '''returns the alpha value associated with gating variable m'''

    return 0.1 * (V + 40) / (1 - np.exp(-0.1 * (V + 40)))

def beta_m(V):
    '''returns the beta value associated with gating variable n'''

    return 4 * np.exp(-0.0556 * (V + 65))

def alpha_h(V):
    '''returns the alpha value associated with gating variable h'''

    return 0.07 * np.exp(-0.05 * (V + 65))

def beta_h(V):
    '''returns the beta value associated with gating variable h'''

    return 1 / (1 + np.exp(-0.1 * (V + 35)))

def get_derivatives(V, m, h, n, i):
    '''retuns the derivatives associated with the quantities V, m, h and n
       in the form of a tuple'''
    
    # current across the ion channels
    i_Na = g_Na * (m**3) * h * (V - E_Na)
    i_K = g_K  * (n**4) * (V - E_K)
    i_L = g_L * (V - E_L)
    i_m = i_Na + i_K + i_L

    #time constant associated with the ion channels
    tau_n = 1 / (alpha_n(V) + beta_n(V))
    tau_m = 1 / (alpha_m(V) + beta_m(V))
    tau_h = 1 / (alpha_h(V) + beta_h(V))

    #the value of the gating variables when time equals infinity
    inf_n = tau_n * alpha_n(V)
    inf_m = tau_m * alpha_m(V)
    inf_h = tau_h * alpha_h(V)

    #derivatives
    dV_dt = (i - i_m) / c_m
    dm_dt = (inf_m - m) / tau_m
    dh_dt = (inf_h - h) / tau_h
    dn_dt = (inf_n - n) / tau_n

    return dV_dt, dm_dt, dh_dt, dn_dt

def run_model(i_e, T, del_t, v_ini, m_ini, h_ini, n_ini):
    ''' runs the Hodgkin-Huxley model and retuns values associated
        with the gating variables and Voltage at a given time'''

    #initialize the variables
    v = v_ini
    m = m_ini
    h = h_ini
    n = n_ini

    t = 0
    t_arr = [t]
    #list containing all the variables
    all_vars = [[v, m, h, n]]

    while round(t,2) < T:
        derivatives = get_derivatives(v, m, h, n, i_e)
        #Euler method to solve the system of ODE's
        v = v + derivatives[0] * del_t
        m = m + derivatives[1] * del_t
        h = h + derivatives[2] * del_t
        n = n + derivatives[3] * del_t
        t += del_t
        all_vars.append([v, m, h, n])
        t_arr.append(t)

    return (t_arr, all_vars)

def compute_period(V_arr, delta_t):
    '''computes the period of the Hodgkin-Huxley model'''

    time_index = []
    n = V_arr.size
    for i in range(1, n):
        if V_arr[i] < 0 and V_arr[i - 1] > 0:
            time_index.append(i)

    if len(time_index) > 1:
        return (time_index[1] - time_index[0]) * delta_t
    else:
        return float('inf')


def run_part_b(ini_values):
    '''correspinds to part b of the assignment, where we plot the spiking 
       rate'''

    i_arr = np.linspace(0, 50, 500) #different values for current
    firing_rate_arr = []
    total_time = 100
    step = 0.05
    for i in i_arr:
        result = run_model(i, total_time, step, ini_values[0], ini_values[1],
            ini_values[2], ini_values[3])
        v_arr = np.asarray(result[1])[:,0]
        period = compute_period(v_arr, 0.05)
        firing_rate_arr.append(1000 / period)

    #plot the results
    fig2 = plt.figure()
    axes2 = fig2.add_subplot(1,1,1)
    axes2.set_xlabel("$I_e$/ A")
    axes2.set_ylabel("Firing Rate (Hz)")
    axes2.set_title("Firing rate of Hodgkin-Huxley model")
    axes2.plot(i_arr, firing_rate_arr)
    plt.show()

def plot_results1(x, y, x_name, y_name, title):
    '''plots a graph for given x and y values'''

    fig = plt.figure()
    axes = fig.add_subplot(1,1,1)
    axes.set_xlabel(x_name)
    axes.set_ylabel(y_name)
    axes.set_title(title)
    axes.plot(x, y)
    plt.show()


def run_part_c(T, step, V_ini, m, h, n):
    '''runs part c of the question, where we plot Hodgkin-Huxley model with 
    a current of -5nA is only supplied for the first 5 ms, and thereafter 
    no current is supplied'''

    #results of the first part of the simulation
    first_part = run_model(-5, 5, step, V_ini, m, h, n)
    time_1 = np.asarray(first_part[0])
    all_vars1 = np.asarray(first_part[1])
    V_arr1 = all_vars1[:, 0]
    m_arr1 = all_vars1[:, 1]
    h_arr1 = all_vars1[:, 2]
    n_arr1 = all_vars1[:, 3]

    v_ini_2 = V_arr1[-1]
    #results of the second part of the simulation, where current supplied is 0
    second_part = run_model(0, 45, step, V_arr1[-1], m_arr1[-1], h_arr1[-1],
                  n_arr1[-1])
    all_vars2 = np.asarray(second_part[1])
    V_arr2 = all_vars2[:, 0]
    m_arr2 = all_vars2[:, 1]
    h_arr2 = all_vars2[:, 2]
    n_arr2 = all_vars2[:, 3]
    time_2 = np.asarray(second_part[0]) + step + 5

    time_comb = np.concatenate((time_1, time_2), axis = None)
    V_comb = np.concatenate((V_arr1, V_arr2), axis = None)
    m_comb = np.concatenate((m_arr1, m_arr2), axis = None)
    h_comb = np.concatenate((h_arr1, h_arr2), axis = None)
    n_comb = np.concatenate((n_arr1, n_arr2), axis = None)

    #plot membrane voltage
    fig = plt.figure()
    axes = fig.add_subplot(1,1,1)
    axes.set_xlabel("time (ms)")
    axes.set_ylabel("V (mv)")
    axes.set_title("Hodgkin-Huxley model when $i_e = -5 \mu A/cm^2 \
        (50 nA/mm^2)$ for t <= 5ms and $i_e$ = 0 for t > 5")
    axes.plot(time_1, V_arr1, label = "For the first 5 ms")
    axes.plot(time_2, V_arr2, label = "For t > 5 ms")
    axes.legend()

    #plot gating probabilities
    fig2 = plt.figure()
    axes2 = fig2.add_subplot(1,1,1)
    axes2.set_xlabel("time (ms)")
    axes2.set_ylabel("m/n/h")
    axes2.set_title("Probabilities assocated with gating variables")
    axes2.plot(time_1, m_arr1, label = "m, when t <= 5")
    axes2.plot(time_2, m_arr2, label = "m, when t > 5")
    axes2.plot(time_1, h_arr1, label = "h, when t <= 5")
    axes2.plot(time_2, h_arr2, label = "h, when t > 5")
    axes2.plot(time_1, n_arr1, label = "n, when t <= 5")
    axes2.plot(time_2, n_arr2, label = "n, when t > 5")
    axes2.legend()
    plt.show()

#constant variables
c_m  =   1.0 #capacitance in uF/cm^2
g_Na = 120.0 #maximal conductance of Na ion channel in mS/cm^2
g_K  =  36.0 #maximal conductance of K ion channel in mS/cm^2
g_L  =   0.3 #maximal conductance of leak channel in mS/cm^2
E_Na =  50.0 #reversal pontenial of Na channel in mV
E_K  = -77.0 #reversal pontenial of K channel in mV
E_L  = -54.387 #reversal pontenial of leak channel in mV
V_0 = -65 # initial membrane potential in mV
i_e = -5 # external current in uA/cm^2

#initial probabilites associated with channel activation(gating variables)
m_0 = 0.0529 
h_0 = 0.5961
n_0 = 0.3177

#time variables
step_size = 0.005
total_time = 50

#part a of the question
results = run_model(i_e,total_time, step_size, V_0, m_0, h_0, n_0)
time_arr = results[0]
variables = np.asarray(results[1])
V = variables[:,0]
plot_results1(time_arr, V, "time (ms)", "V (mV)", "Dynamics of Hodgkin-Huxley\
    Model for $i_e$ = " + str(i_e))

#plot the gating probabilities
fig = plt.figure()
axes = fig.add_subplot(1,1,1)
axes.set_xlabel("time (ms)")
axes.set_ylabel("m/n/h")
axes.set_title("Plot of probabilities associated with gating variables: \
    m, h, and n")
axes.plot(time_arr, variables[:, 1], label = "m")
axes.plot(time_arr, variables[:, 2], label = "h")
axes.plot(time_arr, variables[:, 3], label = "n")
axes.legend()
plt.show()

#part b
run_part_b([V_0, m_0, m_0, h_0, n_0])
#part c of the question
run_part_c(total_time, step_size, V_0, m_0, h_0, n_0)
