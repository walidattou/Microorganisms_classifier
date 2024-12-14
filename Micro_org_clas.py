import matplotlib as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense , Conv2D,MaxPool2D,Flatten,Dropout
import os 
import cv2
import imghdr
import numpy as np

#cleaning data 

data_dir = "C://Users//walid//Desktop//coding projects//Micro org classifier//Micro_Organism"
img_ext = ['jpg','jpeg','png','bmp']

for img_class in os.listdir(data_dir):
    for imgs in os.listdir(os.path.join(data_dir,img_class)):
        image_path = os.path.join(data_dir,img_class,imgs)
        try :
            img = cv2.imread(image_path)
            ext = imghdr.what(image_path)
            if ext in img_ext:
                pass
            else:
                os.remove(image_path)
        except:
            pass


model = Sequential()


data = tf.keras.utils.image_dataset_from_directory(
    "C://Users//walid//Desktop//coding projects//Micro org classifier//Micro_Organism",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
)


data = data.map(lambda x,y :(x/255,y))
data_size = len(data)

#spliting data

train_size = int(data_size*.6)
val_size = int(data_size*.3)
test_size = int(data_size*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#nural network 

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(8, activation='softmax'))

#summary
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
model.summary()

#saving the modele 

hist = model.fit(train,epochs=35,validation_data=val)
model.save(os.path.join('C://Users//walid//Desktop//coding projects//Micro org classifier','ORGCLAS1.h5'))
