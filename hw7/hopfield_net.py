import numpy as np
class Hopfield:

	def __init__(self, N, P, q_0, T):
		'''parameters

		N: dimension of memory vectors used to construct W (weight matrix)
		P: number of n dimensional memory vectors used to construct W
		q_0: initial value of the overlap function(corresponds to probability)
		T: total number of iterations carried out when simulating the model
		'''

		self.N = N
		self.P = P
		self.construct_weight_matrix()
		self.q_0 = q_0
		self.T = T

	def construct_weight_matrix(self):
		'''constructs the weight matrix'''

		weight_matrix = np.zeros(shape = (self.N, self.N))
		memory_vec_li = []
		for i in range(self.P):
			m_1 = np.random.binomial(1, 0.5, self.N)
			m_1 = np.where(m_1 == 0, -1, m_1)
			memory_vec_li.append(m_1)

		for m_a in memory_vec_li:
			m_a.shape = self.N, 1
			weight_matrix += np.dot(m_a, m_a.T)
		for i in range(self.N):
			weight_matrix[i][i] = 0

		self.W = weight_matrix
		self.memory_vecs = memory_vec_li
		self.m = memory_vec_li[0]


	def evaluate_overlap_function(self, s, m):
                '''evaluates the overlap between two vectors s and m'''
		q = 0
		for i in range(self.N):
			q += s[i] * m[i]

		return q / self.N

	def simulate_model(self):
                '''simulates the Hopfield model'''
		S = []
		Q =[]
		s_0 = self.initialize_s()
		Q.append(self.q_0)
		S.append(s_0)
		for t in range(self.T):
			s_1 = np.zeros(self.N)
			for i in range(self.N):
				W_i = self.W[i,:]
				#print(self.W.shape, s_0.shape)
				#print("prod:", W_i * s_0)
				s_1[i] = np.sign(np.sum(W_i * s_0))
			s_0 = s_1
			q_1 = self.evaluate_overlap_function(s_0, self.m)
			#print("t:", t, "s:",s_0, "q:",q_1)
			Q.append(q_1)
			S.append(s_0)
		self.q_f = Q[-1]

	def initialize_s(self):

		s = np.zeros(self.N)
		for i in range(self.N):
			p = np.random.uniform(0,1)
			if p <= self.q_0:
				s[i] = self.m[i]
			else:
				s[i] = 2 * np.random.binomial(1, 0.5, 1) - 1

		return s				








