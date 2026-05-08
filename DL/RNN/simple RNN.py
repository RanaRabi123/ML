import numpy as np
import os
import pandas as pd
import kagglehub
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN

# Download latest version
path = kagglehub.dataset_download("shree0910/ai-vs-human-text-dataset-for-nlp-and-deep-learning")
file_path = os.listdir(path)[0]
df = pd.read_csv(f'{path}/{file_path}')

def data_preprocess():
    text_df = df['text_content']
    # text_df = text_df[:5000]

    tokens = Tokenizer(filters='', lower=False)
    tokens.fit_on_texts(text_df)

    text_seq = tokens.texts_to_sequences(text_df)

    input=[]
    for sentence_vector in text_seq:
        for i in range(1, len(sentence_vector)):
            input.append(sentence_vector[:i+1])

    mas_len = len(max(text_seq, key=len))
    padded_input = pad_sequences(input, maxlen=mas_len, padding='pre')

    X = padded_input[:, :-1]
    y = padded_input[:, -1]
    y_category = to_categorical(y, num_classes=len(tokens.word_index)+1)

    X_tra, X_val, y_tra, y_val = train_test_split(X, y_category, test_size=0.2, random_state=32)

    vocab_size = len(tokens.word_index)+1

    return tokens, mas_len, vocab_size, X_tra, X_val, y_tra, y_val

def build_model():
    model = Sequential([
    Embedding(vocab_size, 64, input_length=(mas_len-1)),
    SimpleRNN(256, return_sequences=False),
    Dense(vocab_size, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_tra, y_tra, batch_size=64, validation_data=(X_val, y_val), epochs=10)

    return model


def sample_with_temperature(predictions, temperature=0.8):
    predictions = np.array(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature 
    predictions = np.exp(predictions)
    predictions = predictions / predictions.sum()          
    return np.random.choice(len(predictions), p=predictions)



def model_prediction(user_input):
    for i in range(30):
        b = tokens.texts_to_sequences([user_input])
        c = pad_sequences(b, maxlen=mas_len, padding='pre')
        prediction = model.predict(c, verbose=0)
        
        pred_index = sample_with_temperature(prediction[0], temperature=1)
        
        if pred_index == 0 or pred_index not in tokens.index_word:
            continue
        
        pred_to_word = tokens.index_word[pred_index]
        user_input += ' ' + pred_to_word
    user_input = user_input.rsplit('.', 1)[0]        # it will remove the last incomplete sentence 

    return user_input








tokens, mas_len, vocab_size, X_tra, X_val, y_tra, y_val = data_preprocess()

model = build_model()


user_input =input('Enter any word to start : ')

print('\t RNN model predition is : \n', model_prediction(user_input), '\n')

