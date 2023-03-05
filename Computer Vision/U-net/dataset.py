import tensorflow as tf
import os

tfds = tf.data.Dataset

class Dataset:

	def __init__(self,train_img_path=None,train_mask_path=None,test_img_path=None,test_mask_path=None,batch_size=100,test_data_only=False):
		self.train_img_path = train_img_path
		self.train_mask_path = train_mask_path
		self.test_img_path = test_img_path
		self.test_mask_path = test_mask_path
		self.batch_size = batch_size
		self.test_data_only = test_data_only

	def get_img_mask(self,img_path,mask_path):
		img = tf.io.read_file(img_path)
		img = tf.image.decode_png(img,channels=3)
		img = tf.image.convert_image_dtype(img,tf.float32)

		mask = tf.io.read_file(mask_path)
		mask = tf.image.decode_png(mask,channels=3)
		mask = tf.math.reduce_max(mask,axis=-1,keepdims=True)
		return img,mask

	def preprocess_img(self,img,mask):
		img = tf.image.resize(img,(800,600))
		mask = tf.image.resize(mask,(800,600))
		return img,mask		

	def load_data(self):
		self.test_img_list = [self.test_img_path+i for i in os.listdir(self.test_img_path)]
		self.test_mask_list = [self.test_mask_path+i for i in os.listdir(self.test_mask_path)]

		self.test_img_fnames = tf.constant(self.test_img_list)
		self.test_mask_fnames = tf.constant(self.test_mask_list)


		self.test_ds = tfds.from_tensor_slices((self.test_img_fnames,self.test_mask_fnames))

		self.test_img_ds = self.test_ds.map(self.get_img_mask)
		self.test_proc_ds = self.test_img_ds.map(self.preprocess_img)

		self.test_ds = self.test_proc_ds.batch(self.batch_size)

		if not self.test_data_only:
			self.train_img_list = [self.train_img_path+i for i in os.listdir(self.train_img_path)]
			self.train_mask_list = [self.train_mask_path+i for i in os.listdir(self.train_mask_path)]
			self.train_img_fnames = tf.constant(self.train_img_list)
			self.train_mask_fnames = tf.constant(self.train_mask_list)
			self.train_ds = tfds.from_tensor_slices((self.train_img_fnames,self.train_mask_fnames))		
			self.train_img_ds = self.train_ds.map(self.get_img_mask)
			self.train_proc_ds = self.train_img_ds.map(self.preprocess_img)
			self.train_ds = self.train_proc_ds.batch(self.batch_size)
			return self.train_ds,self.test_ds
		return self.test_ds	
