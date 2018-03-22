import glob
from PIL import Image
import numpy as np
import pickle
import os.path
from keras.preprocessing import sequence
from keras.applications.inception_v3 import InceptionV3
import json
from keras.preprocessing import image
import pandas as pd

images = 'val2014/'
token = 'annotations/captions_val2014.json'

# Contains all the images
img = glob.glob(images+'*.jpg')

json_data = open(token)
captions = json.load(json_data)
image2id = {}

for i, val in enumerate(captions['images']):
    image2id[val['id']] = val['file_name']

id2captions = {}

for i, val in enumerate(captions['annotations']):
    id2captions[val['image_id']] = val['caption']

image2caption = {}
for key, val in id2captions.items():
    image2caption[image2id[key]] = val


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x

model = InceptionV3(weights='imagenet')
from keras.models import Model

new_input = model.input
hidden_layer = model.layers[-2].output

model_new = Model(new_input, hidden_layer)

def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc

if not os.path.exists('encoded_images_COCO_inceptionV3.p'):
    encoding_train = {}
    for img in img:
        encoding_train[img[len(images):]] = encode(img)
    with open("encoded_images_COCO_inceptionV3.p", "wb") as encoded_pickle:
        pickle.dump(encoding_train, encoded_pickle) 

encoding_train = pickle.load(open('encoded_images_COCO_inceptionV3.p', 'rb'))

caps = []
for key, val in image2caption.items():
    caps.append(' '.join(list(val.split())))

words = [i.split() for i in caps]
unique = []
for i in words:
    unique.extend(i)
unique.extend("<start>".split())
unique.extend("<end>".split())
unique.extend("<unk>".split())
unique = list(set(unique))

with open("unique.p", "wb") as pickle_d:
    pickle.dump(unique, pickle_d) 

f = open('mscoco_training_dataset.txt', 'w')
f.write("image_id\tcaptions\n")

for key, val in image2caption.items():
    f.write(key + "\t" + "<start> " + val +" <end>" + "\n")

f.close()