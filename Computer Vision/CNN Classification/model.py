import tensorflow as tf

def lenet_5(input_shape=(32,32,3)):
    Input = tf.keras.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(filters=16,kernel_size=5,strides=1,padding='same',activation='relu')(Input)
    maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='valid')(conv1)
    conv2 = tf.keras.layers.Conv2D(filters=32,kernel_size=5,strides=1,padding='same',activation='relu')(maxpool1)
    maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=None,padding='valid')(conv2)
    conv3 = tf.keras.layers.Conv2D(filters=120,kernel_size=5,strides=1,padding='same',activation='relu')(maxpool2)  
    flatten = tf.keras.layers.Flatten()(conv3)
    full_cn1 = tf.keras.layers.Dense(240,activation='relu')(flatten)
    full_cn2 = tf.keras.layers.Dense(120,activation='relu')(full_cn1)
    full_cn3 = tf.keras.layers.Dense(4,activation='softmax')(full_cn2)
    m = tf.keras.Model(inputs=Input,outputs=full_cn3)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    m.compile(loss=loss,optimizer='adam',metrics=['accuracy'])
    return m
