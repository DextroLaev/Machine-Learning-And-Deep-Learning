import tensorflow as tf
from dataset import load_data
import matplotlib.pyplot as plt
import numpy as np
import sys

class Linear_Estimator:

	def __init__(self,train_data,train_label,test_data,test_label,weights,biases):
		self.train_data = train_data
		self.train_label = train_label
		self.test_data = test_data
		self.test_label = test_label
		self.weights = tf.Variable(weights,dtype=tf.float32)
		self.biases = tf.Variable(biases,dtype=tf.float32)
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)

	def model(self,input_data):
		return tf.linalg.matmul(input_data,self.weights)+self.biases

	@tf.function
	def update_weights(self,input_d,output):
		loss = lambda : tf.reduce_mean((self.model(input_d)-output)**2)
		self.optimizer.minimize(loss,[self.weights,self.biases])	


	def train(self,epochs,train_data,train_label):
		for ep in range(epochs):
			inputs = train_data
			output = train_label
			self.update_weights(inputs,output)
			if ep%10==0:
				print('\rloss {}'.format(tf.reduce_mean((self.model(inputs)-output)**2),end=''))
				sys.stdout.flush()
		print()
	
	def __plot(self,prediction,test_data,test_label):
		fig = plt.figure()
		ax = fig.add_subplot(111,projection='3d')
		ax.set_xlabel('param1')
		ax.set_ylabel('param2')
		ax.set_zlabel('targets')
		ax.scatter(test_data[:,:1],test_data[:,1:],test_label,c='b',label='ground truth')
		ax.legend()
		ax.scatter(test_data[:,:1],test_data[:,1:],prediction,c='r',label='predictions')
		ax.legend()
		plt.title('Linear regression')
		plt.show()

	def test(self,test_data,test_label):
		predictions = self.model(test_data)
		self.__plot(predictions,test_data,test_label)

def init_model_params():
	weights = tf.ones(shape=[2,1])
	bias = 1
	return (weights,bias)


if __name__ == '__main__':
	epochs = 1000
	(train_data,train_label),(test_data,test_label) = load_data('lg_dataset')
	(weights,bias) = init_model_params()
	estimator = Linear_Estimator(train_data,train_label,test_data,test_label,weights,bias)
	estimator.train(epochs,train_data,train_label)
	estimator.test(test_data,test_label)							