import regex as re
import kagglehub
import os 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras.layers import Embedding, Dense, LSTM , Bidirectional, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping


def data_cleaning(data):
    sentiment = []
    text = []
    for row in data:
        parts = row.split('\t')
        if len(parts) == 3:
            _, label, sentence = parts
            clean_sentence = re.sub(r'https?://\S+|www\.\S+|@\S+', '', sentence)
            sentiment.append(label)
            text.append(clean_sentence)
    return sentiment, text


def data_preprocessing():
    l_e = LabelEncoder()
    sentiment_encode = l_e.fit_transform(sentiment)

    tokens  = Tokenizer(filters='', lower=False)
    tokens.fit_on_texts(text)
    text_sequence = tokens.texts_to_sequences(text)

    mas_len = len(max(text_sequence, key=len))
    vocab_size = len(tokens.word_index)+1

    padded_input= pad_sequences(text_sequence, maxlen=mas_len, padding='post')

    X = padded_input
    Y = to_categorical(sentiment_encode)
    label_count = Counter(sentiment)  # it will count unique number from the list , so that we can pass in the last layer 

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=32)

    return X_train, X_val, Y_train, Y_val, label_count, vocab_size, tokens, mas_len



def glove_processing(vocab_size, tokens):
    embeddings_dict = {}
    with open('glove.6B.50d.txt', 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector


    glove_matrix = np.zeros((vocab_size, 50))

    for word, i in tokens.word_index.items():
        if i < vocab_size:
            vector = embeddings_dict.get(word)
            if vector is not None:
                glove_matrix[i] = vector


    return embeddings_dict, glove_matrix




def frozen_embed(vocab_size, glove_matrix, mas_len, label_count, X_train, Y_train, X_val, Y_val):
    model_froze = Sequential([
            Embedding(input_dim=vocab_size, weights=[glove_matrix], trainable=False, output_dim=50, input_length=mas_len),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dense(len(label_count), activation='softmax')
        ])

    model_froze.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    callback = EarlyStopping(monitor='val_loss', patience=4)
    history_froze = model_froze.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=30, batch_size=64, callbacks=[callback])
    return history_froze






def fine_tune_embed(vocab_size, glove_matrix, mas_len, label_count, X_train, Y_train, X_val, Y_val):
    model_tune = Sequential([
            Embedding(input_dim=vocab_size, weights=[glove_matrix], trainable=True, output_dim=50, input_length=mas_len),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dense(len(label_count), activation='softmax')
        ])

    model_tune.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    callback = EarlyStopping(monitor='val_loss', patience=4)
    history_tune = model_tune.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=30, batch_size=64, callbacks=[callback])
    return history_tune






path = kagglehub.dataset_download("reslanaltinawi/semeval-2017-tweets")

with open(f'{path}/{os.listdir(path)[0]}', 'r') as file:
    data = file.readlines()



sentiment, text = data_cleaning(data)
X_train, X_val, Y_train, Y_val, label_count, vocab_size, tokens, mas_len = data_preprocessing(sentiment, text )
embeddings_dict, glove_matrix = glove_processing(vocab_size, tokens)
history_froze = frozen_embed(vocab_size, glove_matrix, mas_len, label_count, X_train, Y_train, X_val, Y_val)
history_tune = fine_tune_embed(vocab_size, glove_matrix, mas_len, label_count, X_train, Y_train, X_val, Y_val)


print('--'*20)
print("MAX val_accu with forzen embedding is : ", max(history_froze.history['val_accuracy']))
print("MAX val_accu with forzen embedding is : ", max(history_tune.history['val_accuracy']))

print('--'*20)
print("MIN val_loss with forzen embedding is : ", min(history_froze.history['val_loss']))
print("MIN val_loss with forzen embedding is : ", min(history_tune.history['val_loss']))

