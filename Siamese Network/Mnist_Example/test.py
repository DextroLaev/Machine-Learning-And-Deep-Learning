import tensorflow as tf
import matplotlib.pyplot as plt
from model import *

if __name__ == '__main__':
	(train_data,train_label),(test_data,test_label) = tf.keras.datasets.mnist.load_data()

	''' Earlier We trained the model with the number 5, now we will calculate the feature distance between the anchor,positive and negative image
		where negative image will be image representing a number other than 5 ( Here we are taking 4 as negative)'''

	'''so anchor = image of 5, positive = another image of 5, negative = image of 4 ( image other than 5 )'''

	train_data = train_data.reshape(60000,1,28,28)/255
	test_data = test_data.reshape(10000,1,28,28)/255


	anchor = train_data[0]  # image is representing 5
	positive = train_data[11] # another image representing 5
	plt.imshow(anchor[0,:,:])
	plt.show()
	plt.imshow(positive[0,:,:])
	plt.show()
	negative = train_data[2] # image representing 4
	plt.imshow(negative[0,:,:])
	plt.show()

	with tf.keras.utils.CustomObjectScope({'Custom_Distance':Custom_Distance}):
		model = Siamese_Network_Model()
		model.siamese_arch = tf.keras.models.load_model('./model.h5')

	anchor_pos_dis,anchor_neg_dis = model.siamese_arch.predict([anchor,positive,negative])
	print(anchor_pos_dis)
	print(anchor_neg_dis)
	print(anchor_pos_dis<anchor_neg_dis)