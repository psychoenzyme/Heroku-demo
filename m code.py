import tensorflow as tf
from tensorflow import keras
from keras import layers

# define model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# load the data
# train_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_data = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2, 
    horizontal_flip=True)
test_data = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training_set = train_data.flow_from_directory("C:/Users/B DHATRISRI/Desktop/Med_Diag/Datasets/train/", target_size=(150, 150), batch_size=32, class_mode='binary')
test_set = test_data.flow_from_directory("C:/Users/B DHATRISRI/Desktop/Med_Diag/Datasets/test", target_size=(150, 150), batch_size=32, class_mode='binary')

# train the model
model.fit(training_set, epochs=30, validation_data=test_set)

# save the model
model.save('C:/Users/B DHATRISRI/Desktop/Med_Diag/model.h5') 

import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np
# from tensorflow.keras.preprocessing import image
from PIL import Image

# load the model
model = keras.models.load_model('C:/Users/B DHATRISRI/Desktop/Med_Diag/model.h5')

# load the test image
test_image = Image.open('C:\\Users\\B DHATRISRI\\Desktop\\Med_Diag\\Datasets\\test\\Diseased\\0aba288f-aa30-40dc-8ef7-45b03adff020.jpg')
test_image = test_image.resize((150, 150))
# test_image = Image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# make prediction
prediction = model.predict(test_image)

# display the result
if prediction[0][0] < 0.5:
    print("The image is healthy.")
else:
    print("The image is diseased.")

# get the accuracy and loss of the model
accuracy, loss = model.evaluate(test_set)
print("Accuracy:", accuracy)
print("Loss:", loss)
