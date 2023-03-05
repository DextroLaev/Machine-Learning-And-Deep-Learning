import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import Dataset
from model import *

IMG_HEIGHT = 800
IMG_WIDTH = 600
IMG_CHANNELS = 3

callbacks = [
	tf.keras.callbacks.EarlyStopping(monitor='loss',patience=10),
	tf.keras.callbacks.ModelCheckpoint(filepath='self-drive.h5',save_weights_only=True,monitor='loss',save_best_only=True)
]

TRAIN_IMG_PATH = './dataA/CameraRGB/'
TRAIN_MASK_PATH = './dataA/CameraSeg/'
TEST_IMG_PATH = './dataB/CameraRGB/'
TEST_MASK_PATH = './dataB/CameraSeg/'
BATCH_SIZE = 100

if __name__ == '__main__':

	model = init_model(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
	dataset = Dataset(TRAIN_IMG_PATH,TRAIN_MASK_PATH,TEST_IMG_PATH,TEST_MASK_PATH,BATCH_SIZE)
	train_ds,test_ds = dataset.load_data()
	history = model.fit(train_ds.prefetch(tf.data.AUTOTUNE),epochs=50,callbacks=callbacks)
	