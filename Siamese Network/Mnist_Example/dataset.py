import tensorflow as tf
import numpy as np

tfds = tf.data.Dataset

class Dataset:

	def __init__(self):
		(self.train_data,self.train_label),(self.test_data,self.test_label) = tf.keras.datasets.mnist.load_data()
		self.train_data = self.train_data.reshape(60000,1,28,28)/255
		self.test_data = self.test_data.reshape(10000,1,28,28)/255

	def get_anchor_positive(self,number):
		index_anchor_list = np.where(self.train_label==number)[0]
		index_pos_list = np.where(self.test_label==number)[0]

		anchors = []
		positive = []

		total_possible_data = min(len(index_anchor_list),len(index_pos_list))
		for i in range(total_possible_data):
			anchor_img = self.train_data[index_anchor_list[i]]
			positive_img = self.test_data[index_pos_list[i]]
			anchors.append(anchor_img)
			positive.append(positive_img)

		anchors = tf.cast(anchors,tf.float32)
		positive = tf.cast(positive,tf.float32)
		return anchors,positive

	def load_dataset(self,number):
		anchors,positives =	self.get_anchor_positive(number)
		anchor_ds = tfds.from_tensor_slices(anchors)
		pos_ds = tfds.from_tensor_slices(positives)
		negative_images = anchors+positives
		img_count = len(positives)
		neg_ds = tfds.from_tensor_slices(negative_images)
		dataset = tfds.zip((anchor_ds,pos_ds,neg_ds))
		dataset = dataset.shuffle(buffer_size=1024)

		train_ds = dataset.take(int(img_count*0.8))
		val_ds = dataset.skip(int(img_count*0.8))
		train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
		val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
		return train_ds,val_ds