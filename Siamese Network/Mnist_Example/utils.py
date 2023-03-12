import tensorflow as tf
from model import *

tf.config.run_functions_eagerly(True)

class Model(tf.keras.Model):

	def __init__(self,margin = 0.5):
		super().__init__()
		self.margin = margin
		self.siamese_network = Siamese_Network_Model().siamese_arch
		self.loss_tracker = tf.keras.metrics.Mean(name='loss')

	def call(self,inputs):
		return self.siamese_network(inputs)

	def calc_loss(self,data):
		pos_dis,neg_dis = self.siamese_network(data)
		loss = tf.maximum((pos_dis-neg_dis)+self.margin,0)
		return loss

	@tf.function
	def train_step(self,data):
		with tf.GradientTape() as tape:
			loss = self.calc_loss(data)

		gradients = tape.gradient(loss,self.siamese_network.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients,self.siamese_network.trainable_weights))
		self.loss_tracker.update_state(loss)
		return {'loss':self.loss_tracker.result()}

	def test_step(self,data):
		loss = self.calc_loss(data)
		self.loss_tracker.update_state(loss)
		return {'loss':loss}

	@property
	def metrices(self):
		return [self.loss_tracker]	