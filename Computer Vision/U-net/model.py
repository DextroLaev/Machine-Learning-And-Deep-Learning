import tensorflow as tf

def downsampling_block(inputs,n_filters=32,dropout_prob=0,max_pooling=True):
	conv = tf.keras.layers.Conv2D(n_filters,kernel_size=3,activation='relu',padding='same',kernel_initializer='he_normal')(inputs)
	conv = tf.keras.layers.Conv2D(n_filters,kernel_size=3,activation='relu',padding='same',kernel_initializer='he_normal')(conv)
	if dropout_prob > 0:
		conv = tf.keras.layers.Dropout(dropout_prob)(conv)
	if max_pooling:
		next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv)
	else:
		next_layer = conv
	skip_connection = conv
	return next_layer,skip_connection

def upsampling_block(prev_layers,skip_connection,n_filters=32):
	up = tf.keras.layers.Conv2DTranspose(n_filters,kernel_size=3,strides=2,padding='same')(prev_layers)
	merge = tf.keras.layers.concatenate([up,skip_connection],axis=3)
	conv = tf.keras.layers.Conv2D(n_filters,kernel_size=3,activation='relu',padding='same',kernel_initializer='he_normal')(merge)
	conv = tf.keras.layers.Conv2D(n_filters,kernel_size=3,activation='relu',padding='same',kernel_initializer='he_normal')(conv)
	return conv

def Unet_model(input_size=(800,600,3),n_filters=32,n_classes=100):
	inputs = tf.keras.layers.Input(input_size)

	cblock1 = downsampling_block(inputs=inputs,n_filters=n_filters)
	cblock2 = downsampling_block(inputs=cblock1[0],n_filters=n_filters*2)
	cblock3 = downsampling_block(inputs=cblock2[0],n_filters=n_filters*4)
	cblock4 = downsampling_block(inputs=cblock3[0],n_filters=n_filters*8,dropout_prob=0.3)
	cblock5 = downsampling_block(inputs=cblock4[0],n_filters=n_filters*16,dropout_prob=0.3,max_pooling=False)

	ublock6 = upsampling_block(cblock5[0],cblock4[1],n_filters*8)
	ublock7 = upsampling_block(ublock6,cblock3[1],n_filters*4)
	ublock8 = upsampling_block(ublock7,cblock2[1],n_filters*2)
	ublock9 = upsampling_block(ublock8,cblock1[1],n_filters*1)

	conv9 = tf.keras.layers.Conv2D(n_filters,kernel_size=3,activation='relu',padding='same',kernel_initializer='he_normal')(ublock9)
	conv10 = tf.keras.layers.Conv2D(n_classes,1,padding='same')(conv9)
	model = tf.keras.Model(inputs=inputs,outputs=conv10)
	return model

def Init_Model(img_height,img_width,channels):
	unet = Unet_model((img_height,img_width,channels))
	unet.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
	return unet