import glob
from PIL import Image
import numpy as np
import pickle
import os.path
from tqdm import tqdm
from keras.preprocessing import sequence
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image

token = 'Flickr8k_text/Flickr8k.token.txt'
captions = open(token, 'r').read().strip().split('\n')
d = {}
for i, row in enumerate(captions):
    row = row.split('\t')
    row[0] = row[0][:len(row[0])-2]
    if row[0] in d:
        d[row[0]].append(row[1])
    else:
        d[row[0]] = [row[1]]

images = 'Flickr8k_Dataset/Flicker8k_Dataset\\'

# Contains all the images
img = glob.glob(images+'*.jpg')

train_images_file = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

def split_data(l):
    temp = []
    for i in img:
        if i[len(images):] in l:
            temp.append(i)
    return temp

# Getting the training images from all the images
train_img = split_data(train_images)

val_images_file = 'Flickr8k_text/Flickr_8k.devImages.txt'
val_images = set(open(val_images_file, 'r').read().strip().split('\n'))

# Getting the validation images from all the images
val_img = split_data(val_images)

test_images_file = 'Flickr8k_text/Flickr_8k.testImages.txt'
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

# Getting the testing images from all the images
test_img = split_data(test_images)


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

# plt.imshow(np.squeeze(preprocess(train_img[0])))

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

if not os.path.exists('encoded_images_inceptionV3.p'):
    encoding_train = {}
    for img in tqdm(train_img):
        encoding_train[img[len(images):]] = encode(img)
    with open("encoded_images_inceptionV3.p", "wb") as encoded_pickle:
        pickle.dump(encoding_train, encoded_pickle) 

encoding_train = pickle.load(open('encoded_images_inceptionV3.p', 'rb'))

if not os.path.exists('encoded_images_val_inceptionV3.p'):
    encoding_test = {}
    for img in tqdm(val_img):
        encoding_test[img[len(images):]] = encode(img)
    with open("encoded_images_val_inceptionV3.p", "wb") as encoded_pickle:
        pickle.dump(encoding_test, encoded_pickle) 
encoding_test = pickle.load(open('encoded_images_val_inceptionV3.p', 'rb'))

train_d = {}
for i in train_img:
    if i[len(images):] in d:
        train_d[i] = d[i[len(images):]]

val_d = {}
for i in val_img:
    if i[len(images):] in d:
        val_d[i] = d[i[len(images):]]

test_d = {}
for i in test_img:
    if i[len(images):] in d:
        test_d[i] = d[i[len(images):]]

caps = []
for key, val in train_d.items():
    for i in val:
        caps.append('<start> ' + i + ' <end>')

f = open('flickr8k_training_dataset.txt', 'w')
f.write("image_id\tcaptions\n")
for key, val in train_d.items():
    for i in val:
        f.write(key[len(images):] + "\t" + "<start> " + i +" <end>" + "\n")

f = open('flickr8k_validation_dataset.txt', 'w')
f.write("image_id\tcaptions\n")
for key, val in val_d.items():
    for i in val:
        f.write(key[len(images):] + "\t" + "<start> " + i +" <end>" + "\n")