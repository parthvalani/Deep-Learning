# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization, ReLU
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.optimizers import Adam
from keras import initializers
from keras.utils import plot_model, np_utils
from keras import backend as K

# Load dataset
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# Reshaping the inputs
if K.image_data_format() == 'channels_first':
    train_x = train_x.reshape(train_x.shape[0], 3, 32, 32)
    test_x = test_x.reshape(test_x.shape[0], 3, 32, 32)
    input_shape = (3, 32, 32)
else:
    train_x = train_x.reshape(train_x.shape[0], 32, 32, 3)
    test_x = test_x.reshape(test_x.shape[0], 32, 32, 3)
    input_shape = (32, 32, 3)
    
# One-hot encoding
train_y = np_utils.to_categorical(train_y, 10)
test_y = np_utils.to_categorical(test_y, 10)

# Normalizing the data
train_x = np.float32(train_x)
test_x = np.float32(test_x)

train_x = (train_x / 255 - 0.5) * 2
test_x = (test_x / 255 - 0.5) * 2

train_x = np.clip(train_x, -1, 1)
test_x = np.clip(test_x, -1, 1)

########################## Generative model #######################

# latent space dimension
dim = 100
init = initializers.RandomNormal(stddev=0.02)
# building the model
gen_model = Sequential()
gen_model.add(Dense(2*2*512, input_shape=(dim,), kernel_initializer=init))# 2x2x512
gen_model.add(Reshape((2, 2, 512)))
gen_model.add(BatchNormalization())
gen_model.add(LeakyReLU(0.2))

gen_model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same')) # 4x4x256
gen_model.add(BatchNormalization())
gen_model.add(LeakyReLU(0.2))

gen_model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')) # 8x8x128
gen_model.add(BatchNormalization())
gen_model.add(LeakyReLU(0.2))

gen_model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')) # 16x16x64
gen_model.add(BatchNormalization())
gen_model.add(LeakyReLU(0.2))

gen_model.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                              activation='tanh')) # 32x32x3

# Visulization of the generator model
gen_model.summary()

#################### Discriminator model ######################

img_shape = train_x[0].shape
dis_model = Sequential()

dis_model.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                         input_shape=(img_shape), kernel_initializer=init))# 16x16x64
dis_model.add(LeakyReLU(0.2))

dis_model.add(Conv2D(128, kernel_size=5, strides=2, padding='same')) # 8x8x128
dis_model.add(BatchNormalization())
dis_model.add(LeakyReLU(0.2))

dis_model.add(Conv2D(256, kernel_size=5, strides=2, padding='same')) # 4x4x256
dis_model.add(BatchNormalization())
dis_model.add(LeakyReLU(0.2))

dis_model.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))# 2x2x512
dis_model.add(BatchNormalization())
dis_model.add(LeakyReLU(0.2))

dis_model.add(Flatten())

dis_model.add(Dense(1, activation='sigmoid'))

# visualization of discriminator model
dis_model.summary()

# compiling the model
dis_model.compile(Adam(lr=0.0003, beta_1=0.5), loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

#combine the gen_model and the dis_model to make a GAN.


# dis_gen = discriminator(generador(z))
dis_model.trainable = False

input = Input(shape=(dim,))
img = gen_model(input)
output = dis_model(img)
dis_gen = Model(inputs=input, outputs=output)

 # combine compilation of both the model
dis_gen.compile(Adam(lr=0.0004, beta_1=0.5), loss='binary_crossentropy',
            metrics=['binary_accuracy'])

# visulizzation of combined model
dis_gen.summary()

###################### train full model ##################

epochs = 100
batch_size = 32
smooth = 0.1

real = np.ones(shape=(batch_size, 1))
fake = np.zeros(shape=(batch_size, 1))

dis_loss = []
gen_loss = []

for e in range(epochs + 1):
    for i in range(len(train_x) // batch_size):
        
        # Train discriminator model weights
        dis_model.trainable = True
        
        # real samples
        batch_x = train_x[i*batch_size:(i+1)*batch_size]
        dis_loss_real = dis_model.train_on_batch(x=batch_x,
                                                   y=real * (1 - smooth))
        
        # fake Samples
        z = np.random.normal(loc=0, scale=1, size=(batch_size, dim))
        fake_x = gen_model.predict_on_batch(z)
        dis_loss_fake = dis_model.train_on_batch(x=fake_x, y=fake)
         
        # discriminator_model loss
        dis_loss_batch = 0.5 * (dis_loss_real[0] + dis_loss_fake[0])
        
        # Train generator_model weights
        dis_model.trainable = False
        gen_loss_batch = dis_gen.train_on_batch(x=z, y=real)

        print(
            'epoch = {}/{}, batch = {}/{}, dis_loss={}, gen_loss={}'.format(e + 1, epochs, i, len(train_x) // batch_size, dis_loss_batch, gen_loss_batch[0]),
            100*' ',
            end='\r'
        )
    
    dis_loss.append(dis_loss_batch)
    gen_loss.append(gen_loss_batch[0])
    print('epoch = {}/{}, dis_loss={}, gen_loss={}'.format(e + 1, epochs, dis_loss[-1], gen_loss[-1]), 100*' ')

    if e % 10 == 0:
        samples = 10
        fake_x = gen_model.predict(np.random.normal(loc=0, scale=1, size=(samples, dim)))

        for k in range(samples):
            plt.subplot(2, 5, k + 1, xticks=[], yticks=[])
            plt.imshow(((fake_x[k] + 1)* 127).astype(np.uint8))

        plt.tight_layout()
        plt.show()

##########################Evaluate model######################

# plotting the metrics
plt.plot(dis_loss)
plt.plot(gen_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['discriminator', 'Adversarial'], loc='center right')
plt.show()
