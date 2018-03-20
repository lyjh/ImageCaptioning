import glob
from PIL import Image
import numpy as np
import pickle
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
import caption_generator
import os.path

embedding_size = 300
batch_size = 128
max_cap_len = 40
vocab_size = 8256

unique = pickle.load(open('unique.p', 'rb'))

cg = caption_generator.CaptionGenerator()
image_caption_model = cg.create_model()

word2index = cg.word2index
index2word = cg.index2word

image_caption_model.load_weights('weights/weights-improvement-03-3.00.hdf5')

model = InceptionV3(weights='imagenet')
new_input = model.input
hidden_layer = model.layers[-2].output

featurize_model = Model(new_input, hidden_layer)

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

def featurize_image(image):
    image = preprocess(image)
    temp_enc = featurize_model.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc

def predict_captions(img_feat):
    start_word = ["<start>"]
    while True:
        par_caps = [word2index[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_cap_len, padding='post')
        preds = image_caption_model.predict([np.array([img_feat]), np.array(par_caps)])
        word_pred = index2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_cap_len:
            break
            
    return ' '.join(start_word[1:-1])

def beam_search_predictions(img_feat, beam_index = 3):
    start = [word2index["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_cap_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_cap_len, padding='post')
            preds = image_caption_model.predict([np.array([img_feat]), np.array(par_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [index2word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption

def output_result(image):
	img_feat = featurize_image(image)
	Image.open(image).show()
	print ('Normal Max search:', predict_captions(img_feat)) 
	print ('Beam Search, k=3:', beam_search_predictions(img_feat, beam_index=3))
	print ('Beam Search, k=5:', beam_search_predictions(img_feat, beam_index=5))
	print ('Beam Search, k=7:', beam_search_predictions(img_feat, beam_index=7))