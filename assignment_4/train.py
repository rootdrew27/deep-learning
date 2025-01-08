import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path
import numpy as np
import os
from glob import glob
import json

DATA_PATH = Path(r'/data/classes/2024/fall/cs426/roota5351/assignment_4/venv/src/data')
IMG_PATH = DATA_PATH / 'images'
LABEL_PATH = DATA_PATH / 'mask_single_channel'
PLOT_PATH = Path(r'plots/')

BATCH_SIZE = 32
DATA_COUNT = 324
BUFFER_SIZE = DATA_COUNT
RANDOM_STATE = 12535235
tf.random.set_seed(RANDOM_STATE)
IMG_SIZE = (500, 500)
TARG_IMG_SIZE = (256, 256)

fnames = [fname.split('/')[-1] for fname in glob(str(IMG_PATH/'*.png'))]

# create lists of filenames

fname_pairs = sorted([[str(IMG_PATH/fname), str(LABEL_PATH/fname)]
                 for fname in fnames
                ])

# create dataset of filnames

list_ds = tf.data.Dataset.from_tensor_slices(fname_pairs).shuffle(DATA_COUNT, seed=RANDOM_STATE)

# determine splits

test_size = int(DATA_COUNT * 0.3)
val_size = int((DATA_COUNT - test_size) * 0.3)

train_ds = list_ds.skip(test_size)
test_ds = list_ds.take(test_size)

train_ds = train_ds.skip(val_size)
val_ds = train_ds.take(val_size)

def load_img_lbl_pair(fpath_pair):
    img_path, lbl_path = fpath_pair[0], fpath_pair[1]
    img = tf.io.read_file(img_path)
    lbl = tf.io.read_file(lbl_path)
    img, lbl = tf.cast(tf.io.decode_png(img), tf.float32)/255.0, tf.io.decode_png(lbl)
    return img, lbl

# map dataset to loading function

train_ds = train_ds.map(load_img_lbl_pair, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(load_img_lbl_pair, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(load_img_lbl_pair, num_parallel_calls=tf.data.AUTOTUNE)

# define preprocessing and augmentation classes

class Preprocess(tf.keras.layers.Layer):
    def __init__(self, h, w):
        super().__init__()
        self.resize_inputs = keras.layers.Resizing(h, w)
        self.resize_labels = keras.layers.Resizing(h, w, interpolation='nearest')
        
    def call(self, inputs, labels):
        inputs = self.resize_inputs(inputs)
        labels = self.resize_labels(labels)
        return inputs, labels

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.randflip_inputs = keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.randflip_labels = keras.layers.RandomFlip(mode="horizontal", seed=seed)
        
    def call(self, inputs, labels):
        inputs = self.randflip_inputs(inputs)
        labels = self.randflip_labels(labels)
        return inputs, labels

# create batches

TARG_H, TARG_W = TARG_IMG_SIZE[0], TARG_IMG_SIZE[1]

train_batches = (
    train_ds
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .map(Preprocess(h=TARG_H, w=TARG_W))
    .map(Augment(seed=RANDOM_STATE))
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_batches = val_ds.batch(BATCH_SIZE).map(Preprocess(h=TARG_H, w=TARG_W))
test_batches = test_ds.batch(BATCH_SIZE).map(Preprocess(h=TARG_H, w=TARG_W))

# Create Model

# defining Conv2d block for u-net
def Conv2dBlock(inputTensor, numFilters, kernelSize = 3, doBatchNorm = True):
    #first Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, 
                               kernel_size = (kernelSize, kernelSize), 
                               kernel_initializer = 'he_normal', 
                               padding = 'same') (inputTensor)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
    
    x =tf.keras.layers.Activation('relu')(x)
    
    #Second Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, 
                               kernel_size = (kernelSize, kernelSize), 
                               kernel_initializer = 'he_normal', 
                               padding = 'same') (x)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation('relu')(x)
    return x

# Now defining Unet 
def GiveMeUnet(inputImage, output_classes, numFilters = 16, dropouts = 0.1, doBatchNorm = True):
  
    # defining encoder Path
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    p1 = tf.keras.layers.Dropout(dropouts)(p1)
    
    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
    p2 = tf.keras.layers.Dropout(dropouts)(p2)
    
    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
    p3 = tf.keras.layers.Dropout(dropouts)(p3)
    
    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
    p4 = tf.keras.layers.Dropout(dropouts)(p4)
    
    c5 = Conv2dBlock(p4, numFilters * 16, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    # defining decoder path
    u6 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(dropouts)(u6)
    c6 = Conv2dBlock(u6, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u7 = tf.keras.layers.Conv2DTranspose(numFilters*4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(dropouts)(u7)
    c7 = Conv2dBlock(u7, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u8 = tf.keras.layers.Conv2DTranspose(numFilters*2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(dropouts)(u8)
    c8 = Conv2dBlock(u8, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u9 = tf.keras.layers.Conv2DTranspose(numFilters*1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(dropouts)(u9)
    c9 = Conv2dBlock(u9, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    output = tf.keras.layers.Conv2D(output_classes, (1, 1))(c9)
    model = tf.keras.Model(inputs = [inputImage], outputs = [output])
    return model

#instantiate model
output_classes = 6
img_width = 256
img_height = 256
channels = 4
inputs = tf.keras.layers.Input((img_width, img_height, channels)) 
unet = GiveMeUnet(inputs, output_classes, dropouts= 0.07)
unet.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.SparseCategoricalAccuracy())

# only update weights of the last 2 conv layers
for i, layer in enumerate(unet.layers):
    if i == 72: continue
    if i == 75: continue
# train model
EPOCHS = 300
model_history = unet.fit(train_batches,
                        epochs=EPOCHS,
                        validation_data=val_batches,
                        verbose=1)

# save the training history and model weights
hist = model_history.history

with open('hist_1.json', 'w') as f:
    json.dump(hist, f)

unet.save_weights("unet_1.h5") 
