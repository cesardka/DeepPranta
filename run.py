import os
import shutil
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys, getopt

labels = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
# dimensions of our images
img_width, img_height = 64, 64



rootdir = './data/test/'
rootdest = './test/'

model = load_model('new__save__at_33.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


for root, subdirs, files in os.walk(rootdir):
    for file in files:
        original = os.path.join(root, file)
        img = image.load_img(original, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        label = labels[model.predict_classes(images)[0]]
        print(file, label)
        dirname = os.path.join(rootdest, label)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        shutil.copyfile(original, os.path.join(dirname, file))
        
