#The code for assignment 5. The assignment mostly deals with Principal Component Analysis
import numpy as np 
import matplotlib.pyplot as plt 
import numpy.linalg as linalg
import seaborn as sns

def plot_fig(x, y, x_name, y_name, title):

	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	axes.set_xlabel(x_name)
	axes.set_ylabel(y_name)
	axes.set_title(title)
	axes.scatter(x, y)

	plt.show()

def first_question_part_a():

	dim_array = np.linspace(1, 50, 50, dtype = int)
	angle_li = []
	n_trials = 100
	for n in dim_array:
		angle_total = 0
		for i in range(n_trials):
			vec1 = np.random.normal(size = (n, 1))
			vec2 = np.random.normal(size = (n, 1))
			dot = np.dot(vec1.T, vec2)
			cosine_theta = dot[0][0] / (linalg.norm(vec1) * linalg.norm(vec2))
			#print(cosine_theta)
			angle_total += np.arccos(abs(cosine_theta))
		av_angle = angle_total / n_trials
		angle_li.append(av_angle)
	plot_fig(dim_array, angle_li, "dimension", "$\phi$", "Average "\
	                         "angle between two random n dimensional vectors")


def first_question_part_b():

	n_trials = 100
	prob_li = []
	dim_array = np.linspace(1, 50, 50, dtype = int)
	for n in dim_array:
		sum_total = 0
		for i in range(n_trials):
			vec = np.random.uniform(-1, 1, n)
			#print(vec)
			norm = linalg.norm(vec)
			#print(norm)
			if norm < 1:
				sum_total += 1
		prob = sum_total / n_trials
		prob_li.append(prob)

	plot_fig(dim_array, prob_li, "dimension", "Probability", "Probability "\
	                      "that n-dimensional vectors lie within a unit ball")


def second_question_part_a():

	P = 100
	n_dim = 2

	x = np.random.normal(0, 1, P)
	y = np.random.normal(0, 3, P)
	more_points1 = np.linspace(-0.4,0.4,100)
	more_points2 = np.linspace(-4, 4, 100)
	data_points = np.vstack((x,y))

	C = 0
	for i in range(P):
		vec = data_points[:, [i]]
		C += np.dot(vec, vec.T)
	C = C / P
	eig = linalg.eig(C)
	eig_vals, eig_vec = eig
	print(eig_vals)
	print(eig_vec)

	fig1 = plt.figure()
	axes1 = fig1.add_subplot(1,1,1)
	axes1.set_xlabel("x")
	axes1.set_ylabel("y")
	axes1.scatter(x, y, label = "data points", color = "green")
	axes1.set_title("Plot of data points and eigen-vectors")
	axes1.plot(eig_vec[0][0] * more_points2 , eig_vec[1][0] * more_points2, 
		label = "$\lambda$ = " + str(round(eig_vals[1], 4)) + "$e_2$ = (" + str(round(eig_vec[0][0], 4)) + ", " + str(round(eig_vec[1][0], 4))
		       + ")", color = "red")
	axes1.plot(more_points2 * eig_vec[0][1], eig_vec[1][1] * more_points2, 
		label = "$\lambda$ = " + str(round(eig_vals[0], 4)) + " $e_1$ = (" + str(round(eig_vec[0][1], 4)) + ", " + str(round(eig_vec[1][1], 4)) + ")")
	axes1.legend()

	points_eigenspace = np.dot(eig_vec.T, data_points)
	fig2 = plt.figure()
	axes2 = fig2.add_subplot(1,1,1)
	axes2.set_xlabel("x")
	axes2.set_ylabel("y")
	axes2.set_title("Projection in eigenspace")
	axes2.scatter(points_eigenspace[0], points_eigenspace[1])
	plt.show()

def second_question_part_b():

	P = 100
	mean1 = np.asarray([-5, -1])
	mean2 = np.asarray([5, 1])
	sigma = np.asarray([[1, 0], [0, 3]])
	points1 = np.random.multivariate_normal(mean1, sigma, size = P)
	points2 = np.random.multivariate_normal(mean2, sigma, size = P)
	
	C1 = 0
	C2 = 0
	for i in range(P):
		vec1 = points1[[i], :].T
		vec2 = points2[[i], :].T
		C1 += np.dot(vec1, vec1.T)
		C2 += np.dot(vec2, vec2.T)
	eig_val1, eig_vec1 = linalg.eig(C1)
	eig_val2, eig_vec2 = linalg.eig(C2)

	print(eig_val1, eig_vec1)
	print(eig_val2, eig_vec2)



def plot_singular_values(matrix):

	U, S, V_T = linalg.svd(matrix)
	x = np.linspace(1, len(S), len(S))
	plot_fig(x, S, "index of Singular values", "Singular Values", 
		"plot of singular values in descending order")

def plot_eigen_values(matrix):

	covar_mat = np.dot(matrix, matrix.T)
	eig_vals = np.sort(linalg.eig(covar_mat)[0])[::-1] ** 0.5
	l = len(eig_vals)
	n = np.linspace(1, l, l)
	plot_fig(n, eig_vals, "index of eigenvalues", "eigenvalues", 
		"plot of eigenvalues of covariance matrix")

def shuffle_plot_singular_values(matrix):

	np.random.shuffle(matrix)
	U, S, V_T = linalg.svd(matrix)
	n = np.linspace(1, len(S), len(S))
	#plot_fig(n, S, "index of singular values", "Singular Values", 
	#	"Singular values of shuffled data")

	svd_li = []
	for i in range(60):
		np.random.permutation(matrix)
		U_1, S_1, V_T_1 = linalg.svd(matrix)
		svd_li.append(S_1)

	svd_whole = np.asarray(svd_li)
	mean_s = np.mean(svd_whole, axis = 0)
	std_s = np.std(svd_whole, axis = 0)

	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	axes.set_xlabel("index of singular values")
	axes.set_ylabel("Average singular values")
	axes.set_title("Plot of Average singular values")
	axes.scatter(n, mean_s)
	norm_s = linalg.norm(mean_s)
	percent_variance = (mean_s ** 2) / (norm_s ** 2) * 100
	print(norm_s)
	plot_fig(n, percent_variance, "singular values index", "percentage_variance",
		"Percentage of variation explained by singular values")
	print("percent variance", percent_variance)
	print(mean_s)
	plt.show()
	eig_vals = linalg.eig(np.dot(matrix, matrix.T))[0]
	ascend_index = np.argsort(eig_vals)
	plot_fig(np.linspace(0,1,2), np.linspace(1,2,2),"order of singular values",
		"component index","Compontents that stand out(largest singular values")

def third_question():

	location = "/Users/sawal386/Documents/theoretical_neuroscience/"+\
	"assignment_4/data.csv"
	data = np.genfromtxt(location, delimiter = ",")
	plot_singular_values(data)
	plot_eigen_values(data)
	shuffle_plot_singular_values(data)



#first_question_part_a()
#first_question_part_b()
#second_question_part_a()
#second_question_part_b()
third_question()
#eig = np.asarray([[-0.735178, 0.67787], [-0.67787, -0.735179]])
#vec = np.asarray([[0.69, 0.49]])
#plot_singular_values([[1,2,3], [4,5,6]])
#print(np.dot(eig, vec.T))

