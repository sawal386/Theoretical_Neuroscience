#This program implements the Hopfiled network
import numpy as np 
from hopfield_net import Hopfield
import matplotlib.pyplot as plt  

def part_1():

	p_dict = {}
	P_vals = [1, 5, 10]
	q_vals = np.linspace(0,1, 101)
	overall = []
	all_q = []
	for P in P_vals:
		q_sucesss = []
		all_q_final = []
		p_dict[str(P)] = []
		for q in q_vals:
			hopf = Hopfield(100, P, q, 10)
			hopf.simulate_model()
			q_final = hopf.q_f[0]
			p_dict[str(P)].append(q_final)
			all_q_final.append(q_final)
			if q_final > 0.8:
				q_sucesss.append(round(q,2))
		overall.append(q_sucesss)
		all_q.append(all_q_final)

	file = open("/Users/sawal386/Desktop/final.txt", "w")
	for keys in p_dict:
		for vals in p_dict[keys]:
			line  = keys + ", " + str(vals) + "\n"
			file.write(line)

	x_ax = np.linspace(0, 1, 101)		
	file.close()
	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	axes.set_xlabel("q(0)")
	axes.set_ylabel("final q")
	axes.set_title("Plot of final values of q for Hopfield Network")
	axes.scatter(x_ax, p_dict['1'], label = "P = 1")
	axes.scatter(x_ax, p_dict['5'], label = "P = 5")
	axes.scatter(x_ax, p_dict['10'], label = "P = 10")
	axes.legend()
	plt.show()

def part_2():

	P = range(1, 100)
	print(P)
	data = []
	for p in P:
		sum_q = 0
		for i in range(25):
			hopf = Hopfield(100, p, 1, 50)
			hopf.simulate_model()
			sum_q += hopf.q_f
		print(p,sum_q / 25)
		data.append(sum_q / 25)
	fig = plt.figure()	
	axes = fig.add_subplot(1,1,1)
	axes.set_xlabel("P")
	axes.set_ylabel("final_q")
	axes.set_title("Final values of q for a given P")
	axes.scatter(P, data)
	plt.show()

if __name__ == "__main__":
	part_1()
	part_2()


