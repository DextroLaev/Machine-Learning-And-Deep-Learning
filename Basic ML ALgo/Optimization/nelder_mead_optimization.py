import numpy as np

class Nelder_mead_method:

	def __init__(self,dim,function_name,iterations):
		self.dim = dim
		self.function_name = function_name
		self.iterations = iterations
		self.initial_simplex = np.random.randn(dim+1,dim)
		self.functions = {'Himmelblau':self.Himmelblau_func,
						  'Rosenbrock_banana': self.Rosenbrock_func}
		self.function = self.functions[self.function_name]		  

	def sort_simplex(self,simplexes):
		return np.array(sorted(simplexes,key=lambda x:self.function(x)))
	
	def mid_point(self,simplexes):
		mid = []
		for i in range(self.dim):
			coordinate = simplexes[:self.dim,i]
			mid.append(coordinate.sum())	
		mid = (1/self.dim)*np.array(mid)
		return np.array(mid)			

	def distance(self,point1,point2):
		d = []
		for i in range(self.dim):
			d.append(point2[i]-point1[i])
		return np.array(d)

	def operation(self,simplexes,alpha):
		x_c = self.mid_point(simplexes)
		length = self.distance(simplexes[self.dim,:],x_c)
		points = x_c + alpha*length
		return np.array(points) 

	def shrink(self,simplexes):
		for i in range(1,self.dim+1):
			simplexes[i] = simplexes[0] + 0.5*self.distance(simplexes[i],simplexes[0])
		return simplexes	

	def Himmelblau_func(self,param):
		x = param[0]
		y = param[1]
		l = np.square(np.square(x)+y-11) + np.square(x+np.square(y)-7)
		print(x,y,l)
		return l

	def Rosenbrock_func(self,param):
		x = param[0]
		y = param[1]
		# a= 1,b=100
		l = np.square(1-x) + 100*np.square(y-np.square(x))
		print(x,y,l)
		return l

	def minimize(self):
		for i in range(self.iterations):
			self.initial_simplex = self.sort_simplex(self.initial_simplex)
			reflection = self.operation(self.initial_simplex,1)

 			# f(R) < f(W) -> reflection (alpha=1)
			if self.function(reflection) < self.function(self.initial_simplex[-1]):
				expansion = self.operation(self.initial_simplex,2)
				# f(E) < f(R) -> expansion(alpha=2)
				if self.function(expansion) < self.function(reflection):
					self.initial_simplex[-1] = expansion
				else:
					self.initial_simplex[-1] = reflection

			# contraction(alpha=0.5,-0.5)
			elif self.function(reflection) == self.function(self.initial_simplex[-1]):
				outside_contraction = self.operation(self.initial_simplex,0.5)
				inside_contraction = self.operation(self.initial_simplex,-0.5)
				if self.function(outside_contraction) < self.function(inside_contraction):	
					self.initial_simplex[-1] = outside_contraction
				else:
					self.initial_simplex[-1] = inside_contraction

			# shrink		
			else:
				self.initial_simplex = self.shrink(self.initial_simplex)
		return self.initial_simplex[0]		
				
if __name__ == '__main__':
	
	optimizer = Nelder_mead_method(2,'Himmelblau',1000)	
	r = optimizer.minimize()
	print('Minima at : ',r)