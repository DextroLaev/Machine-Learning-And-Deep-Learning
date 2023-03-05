import tensorflow as tf
from dataset import Dataset
import matplotlib.pyplot as plt
from model import Init_Model

IMG_HEIGHT = 800
IMG_WIDTH = 600
IMG_CHANNELS = 3

def load_model(path):
	model = Init_Model(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
	model.load_weights(path)
	return model

def create_mask(pred_mask):
	pred_mask = tf.argmax(pred_mask,axis=-1)
	pred_mask = pred_mask[...,tf.newaxis]
	return pred_mask[0]

def display(display_list):
	fig = plt.figure(figsize=(5,5))
	for i in range(len(display_list)):
		plt.subplot(1,len(display_list),i+1)
		plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
		plt.axis('off')
	plt.show()	

if __name__ == '__main__':
	test_img_path = './dataB/CameraRGB/'
	test_mask_path = './dataB/CameraSeg/'
	model_path = './self-drive.h5'
	test_ds = Dataset(test_img_path=test_img_path,test_mask_path = test_mask_path,test_data_only=True).load_data()
	model = load_model(model_path)
	for img,mask in test_ds.take(6):
		pred_mask = model.predict(img)
		display([img[0],mask[0],create_mask(pred_mask)])
