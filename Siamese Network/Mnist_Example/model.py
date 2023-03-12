import tensorflow as tf

class Custom_Distance(tf.keras.layers.Layer):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)

	def call(self,anchor,positive,negative):
		pos_distance = tf.reduce_sum(tf.square(anchor-positive),axis=-1)
		neg_distance = tf.reduce_sum(tf.square(anchor-negative),axis=-1)
		return (pos_distance,neg_distance)

class Siamese_Network_Model:

	def __init__(self):
		self.embedding = self.base_model()
		self.distance_layer = Custom_Distance()
		self.siamese_arch = self.siamese_network()

	def base_model(self):
		inputs = tf.keras.layers.Input(shape=(28,28,1))
		x = tf.keras.layers.Conv2D(32,kernel_size=2,activation='relu')(inputs)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same')(x)
		x = tf.keras.layers.Conv2D(64,kernel_size=2,activation='relu')(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same')(x)
		x = tf.keras.layers.Conv2D(64,kernel_size=2,activation='relu')(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same')(x)
		x = tf.keras.layers.Conv2D(128,kernel_size=2,activation='relu')(x)
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(1024,activation='softmax')(x)
		m = tf.keras.Model(inputs=[inputs],outputs=[x])
		return m

	def siamese_network(self):
		anchor_img = tf.keras.layers.Input(shape=(28,28,1))
		pos_img = tf.keras.layers.Input(shape=(28,28,1))
		neg_img = tf.keras.layers.Input(shape=(28,28,1))

		distances = self.distance_layer(
				self.embedding(anchor_img),
				self.embedding(pos_img),
				self.embedding(neg_img)
			)

		return tf.keras.Model(inputs=[anchor_img,pos_img,neg_img],outputs=distances)