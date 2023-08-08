import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as image
# Building the AI model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3), activation='relu',input_shape= (150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3),activation= 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])

model.compile(loss= 'binary_crossentropy',
             optimizer=RMSprop(learning_rate = 0.001),
              metrics=['accuracy']
              )

train_datagen = ImageDataGenerator(
                                   # APPLYING IMAGE AUGMENTATION
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    hight_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flib = True,
    fill_mode = 'nearest'
                                   )




val_datagen = ImageDataGenerator (rescale = 1/255)
train_generator = train_datagen.flow_from_directory(
        './horse-or-human/train',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 300x300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
val_generator = val_datagen.flow_from_directory(
        './horse-or-human/validation',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 300x300
        batch_size=16,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')


history  = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data=val_generator,
    validation_steps=8
)

# PLOTTING THE MODEL ACCURACY AND LOSS

plot_history = True
if plot_history ==False:
    pass

else:
    acc = history.history['accuracy']
    val_acc = history.history[ 'val_accuracy']
    loss = history.history[ 'loss' ]
    val_loss = history.history[ 'val_loss']
    epochs = range(len(acc))

    plt.plot (epochs,acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and Validation Accuracy')
    plt.figure()


    plt.plot (epochs, loss)
    plt.plot(epochs,val_loss)
    plt.title('Training and validation loss')
    plt.figure()


images  =os.listdir('test')
for i in images:

    path = 'test/' + i
    print(path)

    img = load_img(path,target_size = (150,150))

    x = img_to_array(img)
    x /=255

    x = np.expand_dims(x,axis=0)

    images = np.vstack([x])
    prediction = model.predict(images,batch_size=10)
    #
    test_image = image.imread(path)
    imgplot = plt.imshow(test_image)
    print(prediction)

    if prediction[0]  > 0.5 :
            plt.title("Human")
            plt.figure()
    elif prediction[0] < 0.5:
            plt.title("Horse")


plt.show()



# PREDICT A SINGLE IMAGE
#
# path = "./test/download.jpeg"
#
#
# img = load_img(path,target_size = (150,150))
#
# x = img_to_array(img)
# x /=255
#
# x = np.expand_dims(x,axis=0)
#
# images = np.vstack([x])
# prediction = model.predict(images,batch_size=10)
#     #
#
# test_image = image.imread(path)
# imgplot = plt.imshow(test_image)
#
# if prediction[0]  > 0.5 :
#             plt.title("Human")
# elif prediction[0] <0.5:
#             plt.title("Horse")
#
# plt.show()