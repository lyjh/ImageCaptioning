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
import os.path

EMBEDDING_DIM = 300

class CaptionGenerator():

    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.index2word = None
        self.word2index = None
        self.total_samples = None
        self.validation_samples = None
        self.use_word_embedding = False
        self.encoded_images = pickle.load( open( "encoded_images_COCO_inceptionV3.p", "rb" ) )
        # self.encoded_val_images = pickle.load( open( "encoded_images_val_COCO_inceptionV3.p", "rb" ) )
        self.variable_initializer()

    def variable_initializer(self):
        df = pd.read_csv('mscoco_training_dataset.txt', delimiter='\t')
        df = df.dropna()
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        for i in range(nb_samples):
            x = next(iter)
            caps.append(x[1][1])

        self.total_samples=0
        for text in caps:
            self.total_samples+=len(text.split())-1
        print ("Total samples : "+str(self.total_samples))

        unique = pickle.load(open('unique_COCO.p', 'rb'))
        self.vocab_size = len(unique)
        self.word2index = {}
        self.index2word = {}
        for i, word in enumerate(unique):
            self.word2index[word]=i
            self.index2word[i]=word

        max_len = 0
        for caption in caps:
            caption_len = len(caption.split())
            if(caption_len > max_len):
                max_len = caption_len
        self.max_cap_len = max_len
        print ("Vocabulary size: "+str(self.vocab_size))
        print ("Maximum caption length: "+str(self.max_cap_len))
        print ("Variables initialization done!")

    def data_generator(self, batch_size = 32, mode='train'):
        partial_caps = []
        next_words = []
        images = []
        
        dataset = 'mscoco_training_dataset.txt' if mode == 'train' else 'mscoco_validation_dataset.txt'
        encode = self.encoded_images if mode == 'train' else self.encoded_val_images
        df = pd.read_csv(dataset, delimiter='\t')
        df = df.dropna()
        df = df.sample(frac=1)
        iter = df.iterrows()
        c = []
        imgs = []
        for i in range(df.shape[0]):
            x = next(iter)
            c.append(x[1][1])
            imgs.append(x[1][0])


        count = 0
        while True:
            for j, text in enumerate(c):
                current_image = encode[imgs[j]]
                for i in range(len(text.split())-1):
                    count+=1
                    
                    partial = [(self.word2index[txt] if txt in self.word2index else self.word2index['<unk>'] )for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    
                    # Initializing with zeros to create a one-hot encoding matrix
                    # This is what we have to predict
                    # Hence initializing it with vocab_size length
                    n = np.zeros(self.vocab_size)
                    # Setting the next word to 1 in the one-hot encoded matrix
                    word = text.split()[i+1]
                    n[self.word2index[word] if word in self.word2index else self.word2index['<unk>']] = 1
                    next_words.append(n)
                    
                    images.append(current_image)

                    if count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
                        count = 0

    def get_or_load_embedding_matrix(self, picklefile, glovefile):
        if not os.path.exists(picklefile):
            embeddings_index = {}
            f = open(glovefile, encoding='utf8')
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

            embedding_matrix = np.zeros((self.vocab_size, EMBEDDING_DIM))
            for word, i in self.word2index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            with open(picklefile, "wb") as encoded_pickle:
                pickle.dump(embedding_matrix, encoded_pickle)
        embedding_matrix = pickle.load(open(picklefile,'rb'))
        return embedding_matrix

    def create_model(self):
        image_model = Sequential([
            Dense(EMBEDDING_DIM, input_shape=(2048,), activation='relu'),
            RepeatVector(self.max_cap_len)
        ])

        if self.use_word_embedding:
            embedding_matrix = self.get_or_load_embedding_matrix('glove_embedding_matrix.p', 'glove.42B.300d.txt')
        
            caption_model = Sequential([
                    Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.max_cap_len, weights=[embedding_matrix], trainable=False),
                    LSTM(256, return_sequences=True),
                    TimeDistributed(Dense(300))
                ])

        else:
            caption_model = Sequential([
                    Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.max_cap_len, trainable=True),
                    LSTM(256, return_sequences=True),
                    TimeDistributed(Dense(300))
                ])
        
        final_model = Sequential([
                Merge([image_model, caption_model], mode='concat', concat_axis=1),
                Bidirectional(LSTM(256, return_sequences=False)),
                Dense(self.vocab_size),
                Activation('softmax')
            ])
        
        final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        final_model.summary()
        return final_model

    def get_word(self,index):
        return self.index_word[index]