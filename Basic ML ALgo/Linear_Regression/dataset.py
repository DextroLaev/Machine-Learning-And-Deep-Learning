import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class lg_dataset:
	def __init__(self):
		self.train_size = 2000
		self.weights = tf.constant([[3],[5]],dtype=tf.float32)
		self.bias = 7
		self.train_param1 = tf.random.uniform([self.train_size],minval=2,maxval=10)
		self.train_param2 = tf.random.uniform([self.train_size],minval=50,maxval=110)
		self.train_data = tf.stack([self.train_param1,self.train_param2],axis=1)
		self.train_label = tf.linalg.matmul(self.train_data,self.weights)+self.bias		

		self.test_param1 = tf.random.uniform([self.train_size],minval=200,maxval=1000)
		self.test_param2 = tf.random.uniform([self.train_size],minval=500,maxval=1100)

		self.test_data = tf.stack([self.test_param1,self.test_param2],axis=1)
		self.test_label = tf.linalg.matmul(self.test_data,self.weights)+self.bias

	def load_data(self):
		return (self.train_data,self.train_label),(self.test_data,self.test_label)

	def plot(self):
		fig = plt.figure()
		ax = fig.add_subplot(111,projection='3d')
		ax.set_xlabel('param1')
		ax.set_ylabel('param2')
		ax.set_zlabel('target')

		ax.scatter(self.test_param1,self.test_param2,self.test_label,c='b',label='ground truth')
		ax.legend()
		plt.show()

class logistic_data:
	def __init__(self):
		self.data = pd.read_csv('data.csv')
		df = self.data.copy()
		df.drop(['competitorname'],axis=1,inplace=True)
		self.independent = df.drop(['hard','sugarpercent','peanutyalmondy','winpercent'],axis=1)
		self.dependent = df['bar']
		self.train_data,self.test_data,self.train_label,self.test_label = train_test_split(self.independent,self.dependent,test_size=0.2)
		self.train_data = tf.Variable(self.train_data,dtype=tf.float32)
		self.test_data = tf.Variable(self.test_data,dtype=tf.float32)
		self.train_label = tf.reshape(tf.Variable(self.train_label,dtype=tf.float32),shape=(-1,1))
		self.test_label = tf.reshape(tf.Variable(self.test_label,dtype=tf.float32),shape=(-1,1))

	def load_data(self):
		return (self.train_data,self.train_label),(self.test_data,self.test_label)

def load_data(dataset_name,plot=False):
	if dataset_name == 'lg_dataset':
		__dataset = lg_dataset()
		if plot:
			__dataset.plot()
		return __dataset.load_data()
	else:
		raise ValueError('dataset not found')

if __name__ == '__main__':
	(train_data,train_label),(test_data,test_label) = load_data('lg_dataset',True)
	print(train_data)
	print(train_label)