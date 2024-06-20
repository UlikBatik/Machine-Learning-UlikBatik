""" Install Mandatory Package(s) """

# Import Deep Learning Library
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import Pretrained Model Library
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import ModelCheckpoint

# Import Dataset Preprocessing Library
import scipy.io
import numpy as np
import pandas as pd

""" Data Preprocessing """

# Define Path Train and Test Directory
train_dataset_path = "./Dataset/train"
test_dataset_path = "./Dataset/test"

BATCH_SIZE = 32

# Make Train and Test Datagen for ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# Generate Image Generator for train and test
train_generator = train_datagen.flow_from_directory(
    directory=train_dataset_path,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    # color_mode='grayscale',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dataset_path,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    # color_mode='grayscale',
    shuffle=True,
    seed=42
)

# Extract a batch of data from the generator
images, labels = next(train_generator)

# Get class indices and create a reverse mapping
images_class_indices = train_generator.class_indices
labels_index_to_class = {v: k for k, v in images_class_indices.items()}

# Define train labels that could be reused on later stage
train_labels = [labels_index_to_class[idx] for idx in range(len(labels_index_to_class))]

""" Model Building & Fitting """

# Define Additional Variabel for Model
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

num_classes = len(train_generator.class_indices)

""" We use pre-trained model VGG19"""

# Load Pretrained VGG19 Model
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Make the Trainable Layers to False
for layer in vgg19.layers:
    layer.trainable = False

# Flatten the output of the VGG16 model
x = tf.keras.layers.Flatten()(vgg19.output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Add a dense layer with ReLU activation
x = tf.keras.layers.Dense(512, activation='relu')(x)

# Add the output layer with softmax activation
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Create the final model
vgg19_model = tf.keras.models.Model(inputs=vgg19.input, outputs=predictions)

EPOCHS = 30

# Compile model
vgg19_model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003),
              metrics=['accuracy']
)

# Define the model checkpoint callback
model_checkpoint = './Checkpoint/model_vgg19_{epoch:02d}_{val_loss:.2f}.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_checkpoint,
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch',
    initial_value_threshold=None
)

# Fit the model with the callback
vgg19_history = vgg19_model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=STEP_SIZE_TEST,
    verbose=1,
    callbacks=[model_checkpoint_callback],
)

# Save model in Keras format
model_save_path = './vgg19_model.h5'
vgg19_model.save(model_save_path)
print("Model saved at:", model_save_path)

# Save training history to CSV
history_df = pd.DataFrame(vgg19_history.history)
history_save_path = './vgg19_history.csv'
history_df.to_csv(history_save_path, index=True)
print("Training history saved at:", history_save_path)