import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image


def predict(temp_file):
    # Load the Xception model
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
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

    # Load the trained weights
    model.load_weights('Model/model_weights.h5')

    # Function to preprocess an image for prediction
    def preprocess_image(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        return img_array

    # Function to make predictions on a single image
    def predict_single_image(img_path):
        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        confidence = predictions[0][predicted_class_index]
        print("predicted_class_index : ", predicted_class_index)
        print("confidence: ",confidence)
        return predicted_class_index, confidence


    return predict_single_image(temp_file)

