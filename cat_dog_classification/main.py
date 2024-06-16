import tensorflow
from tensorflow import keras
from keras.layers import Dense,Flatten
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# modifying the images to overcome the overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# converting the image to float type
train_set = train_datagen.flow_from_directory(
    directory='demo_samples/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
)

test_set = test_datagen.flow_from_directory(
    directory='demo_samples/test',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
)

#load the vgg16 model for transfer learning
vgg_base = VGG16(
    weights='imagenet',
    include_top = False,
    input_shape=(224,224,3)
    )
# vgg_base.summary()
vgg_base.layers[-1].output
flatten = Flatten()(vgg_base.layers[-1].output)
dense1 = Dense(256,activation='relu')(flatten)
dense2 = Dense(128,activation='relu')(dense1)
output = Dense(1,activation='sigmoid')(dense2)

#now combining the vgg16 model input layers & newly created dense(fully conncted layers)
model2_ = Model(inputs=vgg_base.inputs,outputs=output)
# model2_.summary()

#setting the vgg16 layers not to train 
vgg_base.trainable = False
# model2_.summary()

model2_.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model2_.fit_generator(
    train_set,
    epochs=5,
    validation_data = test_set
)

#plotting the results 
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()


