# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
*  Autoencoder is an unsupervised artificial neural network that is trained to copy its input to output. An autoencoder will first encode the image into a lower-dimensional representation, then decodes the representation back to the image.The goal of an autoencoder is to get an output that is identical to the input. Autoencoders uses MaxPooling, convolutional and upsampling layers to denoise the image.
* We are using MNIST Dataset for this experiment. The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.
  
![201967502-00818ac7-4523-46e2-a3be-659758793752](https://github.com/harshavardhini33/convolutional-denoising-autoencoder/assets/93427208/ce80cac1-e4e2-4437-ac35-aeee9f03691d)

## Convolution Autoencoder Network Model


![201967708-8fa56afa-720e-4524-a050-8cbb10b896a5](https://github.com/harshavardhini33/convolutional-denoising-autoencoder/assets/93427208/9bdec539-3ae6-4cd4-837a-99cd6e6d13bd)

## DESIGN STEPS

### STEP 1:
Import the necessary libraries
### STEP 2:
Load the dataset and scale the values for easier computation.
### STEP 3:
Add noise to the images randomly for both the train and test sets.
### STEP 4:
Build the Neural Model using

* Convolutional Layer
* Pooling Layer
* Up Sampling Layer.

### STEP 5:
Pass test data for validating manually.

### STEP 6:
Plot the predictions for visualization.

## PROGRAM

``` python

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x_train.shape

(60000, 28, 28)

x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))
x=layers.Conv2D(16,(3,3),padding='same',activation='relu')(input_img)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),padding='same',activation='relu')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(4,(3,3),padding='same',activation='relu')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(4,(3,3),padding='same',activation='relu')(x)
encoder_layer=layers.MaxPooling2D((2,2),padding='same')(x)

x=layers.Conv2D(8,(3,3),padding='same',activation='relu')(encoder_layer)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(3,3),padding='same',activation='relu')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(3,3),padding='same',activation='relu')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(3,3),padding='same',activation='relu')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(5,5),activation='relu')(x)
x=layers.UpSampling2D((1,1))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = keras.Model(input_img, decoded)

autoencoder.summary()

Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                 
 conv2d_3 (Conv2D)           (None, 28, 28, 32)        320       
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 14, 14, 32)       0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 14, 14, 32)        9248      
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 32)         0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 7, 7, 32)          9248      
                                                                 
 up_sampling2d (UpSampling2D  (None, 14, 14, 32)       0         
 )                                                               
                                                                 
 conv2d_6 (Conv2D)           (None, 14, 14, 32)        9248      
                                                                 
 up_sampling2d_1 (UpSampling  (None, 28, 28, 32)       0         
 2D)                                                             
                                                                 
 conv2d_7 (Conv2D)           (None, 28, 28, 1)         289       
                                                                 
=================================================================
Total params: 28,353
Trainable params: 28,353
Non-trainable params: 0
_________________________________________________________________

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))

Epoch 1/2
469/469 [==============================] - 153s 324ms/step - loss: 0.1643 - val_loss: 0.1186
Epoch 2/2
469/469 [==============================] - 147s 313ms/step - loss: 0.1148 - val_loss: 0.1103

decoded_imgs = autoencoder.predict(x_test_noisy)

313/313 [==============================] - 6s 20ms/step

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![238985896-9c7e3c41-7c3f-4efa-a201-0fa8ae7d66e0](https://github.com/harshavardhini33/convolutional-denoising-autoencoder/assets/93427208/a17c89c3-7b02-484a-a5ef-4118041d7dc8)

### Model Summary
![238986078-7e32f619-a652-4b56-aab5-af5b34457ac6](https://github.com/harshavardhini33/convolutional-denoising-autoencoder/assets/93427208/4f04b8a3-863a-4f81-9279-04d67dd6ff17)


### Original vs Noisy Vs Reconstructed Image

![238985873-45cf671b-5cc4-4b80-98f1-67a0940adc45](https://github.com/harshavardhini33/convolutional-denoising-autoencoder/assets/93427208/c1b4feaf-29aa-4b84-8f51-3ab02d74aac5)



## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.
