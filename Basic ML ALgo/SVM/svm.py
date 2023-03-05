import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class lg_dataset:
	def __init__(self):
		self.train_size=2000
		self.train_param1=tf.random.uniform([100],minval=2,maxval=10)
		self.train_param2=tf.random.uniform([100],minval=50,maxval=110)
		self.train_data=tf.stack([self.train_param1,self.train_param2],axis=1)
		self.test_size=200
		self.train_param11=tf.random.uniform([100],minval=12,maxval=20)
		self.train_param22=tf.random.uniform([100],minval=60,maxval=120)
		self.train_data = tf.concat([self.train_data,tf.stack([self.train_param11,self.train_param22],axis=1)],0)
		self.train_label = tf.random.uniform([100],minval=1,maxval=5)
		self.train_label = tf.concat([self.train_label,tf.random.uniform([100],minval=-1,maxval=-5)],0)

	def load_data(self):
		return (self.train_data,self.train_label)
	
	def plot(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')	
		ax.set_xlabel('param1')	
		ax.set_ylabel('param2')
		ax.set_zlabel('targets')
		ax.scatter(self.train_data[:,0],self.train_data[:,1],self.train_label,c='g',label="ground truth")
		ax.legend()	
		plt.show()

def load_data(dataset_name,plot=False):
	if dataset_name=='svm':
				__dataset=lg_dataset()
				if plot:
					__dataset.plot()
				return __dataset.load_data()
	else:
		raise ValueError('Dataset not found')

class SVM:
	def __init__(self,train_data,train_label,epochs):
		self.train_data = train_data
		self.train_label = train_label
		self.initialize_parameter()
		self.weights = tf.Variable(self.weights,dtype=tf.float32)
		self.bias = tf.Variable(self.bias,dtype=tf.float32)
		self.epsilon = 1
		self.epochs = epochs
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

	def model(self,data):
		return tf.linalg.matmul(data,self.weights)+self.bias

	@tf.function	
	def update_param(self,data,label):
		self.loss = lambda :  tf.reduce_sum(0.5*((self.weights)**2) + tf.reduce_sum(tf.maximum(0.0,self.epsilon-(label*self.model(data)))))
		self.optimizer.minimize(self.loss,var_list=[self.weights,self.bias])

	def train(self):
		data = self.train_data
		label = self.train_label
		while self.epochs:
			self.update_param(data,label)
			self.epochs -= 1

	def plot3D(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		clas_1 = self.model(self.train_data)
		ax.set_xlabel('x1')
		ax.set_ylabel('x2')
		ax.set_zlabel('targets')
		ax.scatter(self.train_data[:,0],self.train_data[:,1],self.train_label,c='g',label="train_data")
		ax.scatter(self.train_data[:,0],self.train_data[:,1],clas_1-1,c='r',label='boundary > 1')
		ax.scatter(self.train_data[:,0],self.train_data[:,1],clas_1,c='b',label = 'boundary = 0')
		ax.scatter(self.train_data[:,0],self.train_data[:,1],clas_1+1,c='r',label='boundary < 1')
		ax.legend()
		plt.show()

	def plot2D(self):
		plt.scatter(self.train_data[:,0],self.train_data[:,1],c='g',label='test_data')
		plt.xlabel('x1')
		plt.ylabel('x2')
		plt.legend()
		plt.plot()	
		plt.show()

	def plot_data(self):
		fig = plt.figure()
		ax = fig.add_subplot(111,projection='3d')
		ax.set_xlabel('x1')
		ax.set_ylabel('x2')
		ax.set_zlabel('targets')
		ax.scatter(self.train_data[:,0],self.train_data[:,1],self.train_label,c='g',label="train_data")
		ax.legend()
		plt.show()	
		
	def initialize_parameter(self):
		self.weights = tf.ones([self.train_data.shape[1],1],dtype=tf.float32)
		self.bias = [0]	

if __name__=='__main__':
	(train_data,train_label) = load_data('svm')
	epochs = int(input("Enter Epochs:- "))
	classifier = SVM(train_data,train_label,epochs)
	tf.print('width: ',2/tf.norm(classifier.weights))
	classifier.plot_data()
	classifier.train()
	tf.print('width: ',2/tf.norm(classifier.weights))
	classifier.plot3D()
