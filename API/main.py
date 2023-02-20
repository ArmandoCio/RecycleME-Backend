import numpy as np
from flask import Flask, request
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Load the saved machine learning model
model = tf.keras.models.load_model('save_at_17.h5')

class_names = ["cardboard", "glass", "metal",  "plastic"]

@app.route('/predict', methods=['POST'])
def predict():
    # Read the image from the request
    image = request.files['image'].read()

    # Decode the JPEG image
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (512, 384))

    # Add the batch dimension
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)  # Create batch axis

    prediction = model.predict(image_array)
    print(prediction[0])
    test = np.argmax(prediction[0])
    print(class_names[test])


    # Return the class name
    return class_names[test]

if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
