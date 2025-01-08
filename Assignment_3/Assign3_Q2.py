import tensorflow as tf
from tensorflow import keras
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 32
RANDOM_STATE = 12535235
DATA_PATH = Path('/data/users/roota5351/cs426_course_files/UCMerced_Dataset/Images')
NEW_DATA_PATH = Path('./data/')
PLOT_PATH = Path('./plots/')
IMG_SIZE = (244, 244)
tf.random.set_seed(RANDOM_STATE)

# Set an environment variable for the data directory
print(tf.config.list_physical_devices())
def load_datasets(from_data_path:Path, to_data_path:Path):
    # check if to_data_path is not empty
    if len(os.listdir(to_data_path)) < 2:
        for dir in os.listdir(from_data_path):
            os.makedirs(os.path.join(to_data_path, dir), exist_ok=True)
            for file in os.listdir(os.path.join(from_data_path, dir)):
                img = keras.utils.load_img(os.path.join(from_data_path, dir, file), target_size=IMG_SIZE)
                keras.utils.save_img(os.path.join(to_data_path, dir, file.replace('.tif', '.png')), img) # save the image as a png

    return tf.keras.preprocessing.image_dataset_from_directory(
        to_data_path,
        image_size=IMG_SIZE,
        validation_split=0.3,
        subset='both',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_STATE,
    )

# create a dictionary to map the labels to the class names
class2int = {dir: i for i, dir in enumerate(os.listdir(DATA_PATH))}
train_dataset, test_dataset = load_datasets(DATA_PATH, NEW_DATA_PATH)

# data augmentation
augs = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.3),
])
# create VGG model
vgg16 = keras.applications.VGG16( # NOTE: The Input Size was 224x224 during training
    input_shape=IMG_SIZE+(3,),
    include_top=False,
    weights='imagenet',
)

for layer in vgg16.layers:
    layer.trainable = False

# Modify the layers of vgg16 by adding a regularizer to each (to prevent overfitting)
for layer in vgg16.layers:
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer = keras.regularizers.l1(0.01)

preprocess_input = keras.applications.vgg16.preprocess_input #  converts images to BGR and zero-centers each color channel

inputs = tf.keras.Input(shape=IMG_SIZE+(3,))
x = augs(inputs)
x = preprocess_input(x)
#x = rescale(x)
x = vgg16(x, training=False)
x = keras.layers.Flatten()(x)
x = keras.layers.Dropout(0.2)(x) # add dropout to prevent overfitting
x = keras.layers.Dense(128, activation='relu')(x)
outputs = keras.layers.Dense(len(class2int.keys()))(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, clipvalue=1.0),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)

history = model.fit(train_dataset, epochs=200, validation_data=test_dataset, verbose=2)

# confusion matrix
y_pred = model.predict(test_dataset)
y_pred = tf.argmax(y_pred, axis=1)

_, labels = tuple(zip(*test_dataset))

labels_np = np.array([])
for arr in labels:
    labels_np = np.concatenate([labels_np, arr])

# confusion matrix
cm = confusion_matrix(labels_np, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class2int.keys())
disp.plot()
plt.savefig(PLOT_PATH/'a3q2_confusion_matrix.png')

plt.clf()
# plot training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig(PLOT_PATH/'a3q2_accuracy.png')

plt.clf()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.savefig(PLOT_PATH/'a3q2_loss.png')