import keras
from tensorflow.keras.preprocessing import image
import matplotlib as plt
import numpy as np

img = image.load_img('oui.jpg',target_size=(200,200))

ml= keras.models.load_model('model.h5')

X = image.img_to_array(img)
X = np.expand_dims(X,axis=0)
images = np.vstack([X])

print(ml.predict(images))

