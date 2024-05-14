import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'skin-disease-datasaet/train_set',
    target_size=(224, 224),
    batch_size=2,  # Adjust the batch size based on your preference
    class_mode='categorical'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_directory(
    'skin-disease-datasaet/test_set',
    target_size=(224, 224),
    batch_size=2,  # Adjust the batch size based on your preference
    class_mode='categorical'
)


# Load Xception model
base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="xception_weights_tf_dim_ordering_tf_kernels_notop.h5",  # You can use imagenet weights
    input_shape=(224, 224, 3),
)

# Make layers of the base model untrainable
for layer in base_model.layers:
    layer.trainable = False

# Define the input and output layers of the base model
base_input = base_model.layers[0].input
base_output = base_model.layers[-1].output

# Add output layers to the base model
l1 = Flatten()(base_output)
l2 = Dense(512, activation='elu')(l1)
l3 = Dropout(0.25)(l2)
l4 = Dense(8, activation='softmax')(l3)

model = tf.keras.Model(inputs=base_input, outputs=l4)

# Compile the model
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_generator.classes), 
    y=train_generator.classes
)

class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)
# Get the class indices from the training generator
class_indices = train_generator.class_indices
print(class_indices)
# Reverse the mapping to get a dictionary of class names
class_names = {v: k for k, v in class_indices.items()}

# Print or use the class names as needed
print("Class Names:", class_names)  

input()
# Class Names: {0: 'BA- cellulitis', 1: 'BA-impetigo', 2: 'FU-athlete-foot', 3: 'FU-nail-fungus', 4: 'FU-ringworm', 5: 'PA-cutaneous-larva-migrans', 6: 'VI-chickenpox', 7: 'VI-shingles'}

# Set up the callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    callbacks=[checkpoint_cb, early_stopping_cb],
    epochs=10,  # You can experiment with the number of epochs
    class_weight=class_weight_dict,
    verbose=1
)

# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# 
# Serialize weights to HDF5
model.save_weights("model.h5")

# Print training history
print("Training History:")
print(history.history)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
