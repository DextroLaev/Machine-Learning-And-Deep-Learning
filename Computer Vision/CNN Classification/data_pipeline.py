import tensorflow as tf
import os

tfds = tf.data.Dataset

class Tensorflow_dataset:

	def __init__(self,path,batch_size=100):
		self.train_path = path+'/train/*/*'
		self.test_path = path+'/test/*'
		self.batch_size = batch_size

	def get_label(self,path):
		path_label = path_label = tf.strings.split(path,sep=' ',maxsplit=1)[1]
		label = tf.strings.to_number(tf.strings.split(path_label,'/',maxsplit=1)[0],tf.int32)
		return label-1	

	def get_image(self,file):
		img = tf.io.read_file(file)	
		decode_img = tf.io.decode_png(img,channels=3)
		img = tf.image.resize(decode_img,[32,32])
		return img

	def parse_func(self,files,data_type):
		img = []
		label = []
		if data_type == 'test':
			path_label = tf.strings.split(files,' ',maxsplit=1)[1]
			label.append(tf.strings.to_number(tf.strings.split(path_label,'.')[0],tf.int32))
			img.append(self.get_image(files))
		else:
			for i in range(len(files)):
				label.append(self.get_label(files[i]))
				img.append(self.get_image(files[i]))
		return img,label
	
	def preprocess_img(self,img,label):
		return img/255.0,label

	def train_gen(self):
		for img,label in self.train_img:
			yield img,label

	def test_gen(self):
		for img,label in self.test_img:
			yield img,label

	def load_images_gen(self):
		train_files = tfds.list_files(self.train_path,shuffle=True)
		test_files = tfds.list_files(self.test_path)
		train_ds_batch = train_files.batch(self.batch_size,drop_remainder=True)
		self.train_img = train_ds_batch.map(lambda x:self.parse_func(x,'train'))
		self.test_img = test_files.map(lambda x:self.parse_func(x,'test'))
		self.train_img.map(lambda x,y:self.preprocess_img(x,y))
		self.test_img.map(lambda x,y:self.preprocess_img(x,y))

		train_gen = tfds.from_generator(
			generator = self.train_gen,
			output_signature = (
				tf.TensorSpec(shape=(None,32,32,3),dtype=tf.float32),
				tf.TensorSpec(shape=(100,),dtype=tf.int32))).prefetch(tf.data.AUTOTUNE)
		test_gen = tfds.from_generator(
        	generator=self.test_gen,
        	output_signature=(
        	tf.TensorSpec(shape=(None,32,32,3), dtype=tf.float32),
        	tf.TensorSpec(shape=(1,), dtype=tf.int32))).prefetch(tf.data.AUTOTUNE)
		return train_gen,test_gen
