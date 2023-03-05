import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import neural_structured_learning as nsl

(train_data,train_label),(test_data,test_label) = tf.keras.datasets.mnist.load_data()

class CustomCallback(tf.keras.callbacks.Callback):
	def __init__(self,func,data,label):
		self.data = tf.cast(data,dtype=tf.float32)
		self.label = tf.cast(label,dtype=tf.int32)
		self.func = func

	def on_batch_end(self,epoch,logs=None):
		perbutaions = self.func(self.data,self.label)
		new_data = self.data + 0.02*perbutaions
		self.model.Input = new_data

class Model:
	
	def __init__(self):
		self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
		self.model = self.mode_arch()
	
	def model_arch(self,input_shape=(28,28)):
		Input = tf.keras.Input(input_shape)
		flatten = tf.keras.layers.Flatten()(Input)
		hd1 = tf.keras.layers.Dense(240,activation='relu')(flatten)
		hd2 = tf.keras.layers.Dense(120,activation='relu')(hd1)
		out = tf.keras.layers.Dense(10,activation='softmax')(hd2)
		m = tf.keras.Model(inputs=Input,outputs=out)
		m.compile(loss=self.loss_fn,optimizer='adam',metrics=['accuracy'])
		return m

	def create_noise(self,image,label):
		with tf.GradientTape() as tape:
			tape.watch(image)
			pred = self.model(image)
			loss = self.loss_fn(label,pred)

		gradient = tape.gradient(loss,image)
		signed_grad = tf.sign(gradient)
		return signed_grad

	def train(self,train_data,train_label,test_data,test_label):
		self.model.fit(train_data,train_label,validation_data=(test_data,test_label),epochs=10,callbacks=[CustomCallback(self.create_noise,train_data,train_label)])

model = tf.keras.Sequential([
    tf.keras.Input((28, 28), name='feature'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

if __name__ == '__main__':
	m = Model()
	# neural_structured_learning
	adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2,adv_step_size=0.05)
	adv_model = nsl.keara.AdversarialRegularization(model,adv_config=adv_config)
	adv_model.compile(optimizer='adam',
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])

	adv_model.fit({'feature':train_data,'label':train_label},batch_size=32,epochs=5)
	adv_model.evaluate({'feature':test_data,'label':test_label})
