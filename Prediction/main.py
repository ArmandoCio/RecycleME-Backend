import tensorflow as tf
from tensorflow import keras
import numpy as np
import PIL
import cv2


num_classes = ["cardboard", "glass", "metal", "paper", "plastic"]

# define a video capture object
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('Recycle ME!', frame)

    # the 'q' button is set as the
    # quitting button you may use q
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # After the loop release the cap object
        # Destroy all the windows
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(1) & 0xFF == ord('/'):
        cv2.imwrite("test.jpg", frame)
        vid.release()
        cv2.destroyAllWindows()
        break



image_size = (512, 384)

reconmodel=keras.models.load_model("../pythonProject8/save_at_6.h5")


img = keras.preprocessing.image.load_img(
    "test.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = reconmodel.predict(img_array)
print(predictions[0])
test=np.argmax(predictions[0])
print(num_classes[test])