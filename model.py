from keras.layers import *
from keras.models import *
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

def get_model():

    input = Input(shape=(32,32,1))
    x = Conv2D(32, 3, activation='relu', padding='same')(input)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(1, 3, activation=None, padding='same')(x)
    x = Activation('tanh')(x)
    x = x*127.5+127.5

    model = Model([input], x)
    model.summary()
    return model 

def get_data(path):
    x = []
    y = []
    for img_dir in tqdm(glob('DIV2K_train_HR\*.png')):
        img = cv2.imread(img_dir)
        img_ycrb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel = img_ycrb[:,:,1]

        y_out = cv2.resize(y_channel,(128,128),interpolation=cv2.INTER_AREA)
        y_in = cv2.resize(y_out, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
        x.append(y_in)
        y.append(y_out)
    x = np.array(x)
    y = np.array(y)

    return x,y


model = get_model()
curr_path = os.getcwd()
curr_path = curr_path + '\DIV2K_train_HR\*.png'
x, y = get_data(curr_path)
print(x.shape,y.shape)

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss = 'mse'
model.compile(loss=loss, optimizer=optimizer)

save_model_callback = tf.keras.callbacks.ModelCheckpoint(
    'model/model_u_channel.h5',
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    mode = 'min',
    save_freq = 'epoch'
)

tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
batch_size = 4
epochs = 100
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tbCallBack, save_model_callback])



