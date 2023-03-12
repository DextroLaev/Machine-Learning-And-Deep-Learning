import tensorflow as tf
from dataset import *
from utils import *

if __name__ == '__main__':
	train_ds,val_ds = Dataset().load_dataset(5)

	siamese_model = Model()
	siamese_model.compile(optimizer=tf.keras.optimizers.Adam())

	siamese_model.fit(train_ds,epochs=1,validation_data=val_ds)
	siamese_model.siamese_network.save('./model.h5')