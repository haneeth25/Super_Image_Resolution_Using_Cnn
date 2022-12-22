import tensorflow as tf
import os 
import cv2

import numpy as np
import PIL
from PIL import Image
DirPath_lr_data = '/content/drive/MyDrive/SR MODEL/low_res'
DirPath_hr_data = '/content/drive/MyDrive/SR MODEL/high_res'
img_names = os.listdir(DirPath_lr_data)
train__lr = []
train__hr= []

for each_img in img_names:
  if each_img != '.DS_Store':
    img_path = os.path.join(DirPath_lr_data,each_img)
    img = PIL.Image.open(img_path)
    img_arr = np.array(img) 
    train__lr.append(img_arr)

for each_img in img_names:
  if each_img != '.DS_Store':
    img_path = os.path.join(DirPath_hr_data,each_img)
    img = PIL.Image.open(img_path)
    img_arr = np.array(img) 
    train__hr.append(img_arr)

train__lr = np.array(train__lr)
train__lr = train__lr/255.
train__lr[0]

train__hr = np.array(train__hr)
train__hr = train__hr/255.
train__hr[0]
     
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dense,Flatten,Dropout,UpSampling2D,Add
from tensorflow.keras import regularizers
from keras.optimizers import Adam 
from keras.models import Model
     
def Super_Resolution():
    input_img = Input(shape=train__lr.shape[])
    
    l1 = Conv2D(64,kernel_size=4,activation="relu",padding="same",activity_regularizer=regularizers.l1(10e-10))(input_img)
    l1 = Conv2D(64,kernel_size=4,activation="relu",padding="same",activity_regularizer=regularizers.l1(10e-10))(l1)
    m1 = MaxPooling2D(pool_size=2)(l1)

    l2 = Conv2D(128,kernel_size=4,activation="relu",padding="same",activity_regularizer=regularizers.l1(10e-10))(m1)
    l2 = Conv2D(128,kernel_size=4,activation="relu",padding="same",activity_regularizer=regularizers.l1(10e-10))(l2)
    m2 = MaxPooling2D(pool_size=2)(l2)
    
    l3 = Conv2D(256,kernel_size=4,activation="relu",padding="same",activity_regularizer=regularizers.l1(10e-10))(m2)
    
    u1 = UpSampling2D(size=2)(l3)
    l4 = Conv2D(128,kernel_size=4,activation="relu",padding="same",activity_regularizer=regularizers.l1(10e-10))(u1)
    l4 = Conv2D(128,kernel_size=4,activation="relu",padding="same",activity_regularizer=regularizers.l1(10e-10))(l4)
    add1 = Add()([l4,l2])

    u2 = UpSampling2D(size=2)(add1)
    l5 = Conv2D(64,kernel_size=4,activation="relu",padding="same",activity_regularizer=regularizers.l1(10e-10))(u2)
    l5 = Conv2D(64,kernel_size=4,activation="relu",padding="same",activity_regularizer=regularizers.l1(10e-10))(l5)
    add2 = Add()([l5,l1])

    output = Conv2D(4,kernel_size=4,activation="linear",padding="same",activity_regularizer=regularizers.l1(10e-10))(add2)
    
    model = Model(input_img,output)
    return model

model2 = Super_Resolution()
model2.summary()


model2.compile(loss="mean_squared_error",optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])

model2.fit(train__lr,train__hr,epochs=100,batch_size=32,verbose=1)

img = PIL.Image.open('/content/442.png')
img_arr = np.array(img) 

input_to_model = np.array([img_arr])

test_img = img_arr/255.
input_to_model = np.array([test_img])
input_to_model.shape

  

hr_output = model2.predict(input_to_model)


arr = np.squeeze(hr_output) # you can give axis attribute if you wanna squeeze in specific dimension
plt.imshow(arr)
plt.show()
     

img_orginal = PIL.Image.open('/content/442_high.png')
img_orginal_arr = np.array(img_orginal) 
     
from math import log10, sqrt
import cv2
import numpy as np
  
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

v1 = PSNR(img_orginal_arr/255.,img_arr/255.)
v1

v2 = PSNR(img_orginal_arr/255.,hr_output)
v2
   
