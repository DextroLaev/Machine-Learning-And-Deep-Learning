import tensorflow as tf
import numpy as np
from data_pipeline import Tensorflow_dataset
from model import lenet_5

if __name__ == '__main__':
	data = Tensorflow_dataset('Dataset')
	train_gen,test_gen = data.load_images_gen()

	model = lenet_5()
	model.fit(train_gen,epochs=10)

	output = tf.argmax(model.predict(test_gen),axis=1)
	print(output)


