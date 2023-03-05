import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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

def load_data(dataset_name):
	if dataset_name == 'logistic_regression':
		__dataset = logistic_data()
		return __dataset.load_data()	
	else:
		raise ValueError('dataset not found')

if __name__ == '__main__':
	(train_data,train_label),(test_data,test_label) = load_data('logistic_regression')
	print(train_data)
	print(train_label)