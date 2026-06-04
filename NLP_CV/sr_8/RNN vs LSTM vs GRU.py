import kagglehub
import os
import json
from keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import Counter
from keras.layers import SimpleRNN, Embedding, Dense, LSTM , GRU

class models():
    def data_cleaning(self):
        self.text, self.compliment_count = [], []

        for i in data:
            temp = {
                "text": None,
                "compliment_count": None
                }

            for key, value in i.items():
                if value is None:
                    continue

                if key == "text":
                    temp["text"] = value
                elif key == "compliment_count":
                    temp["compliment_count"] = value

            if temp["text"] is not None:            # only append if text exists, so that length don;t mismatch
                self.text.append(temp["text"])
                self.compliment_count.append(temp["compliment_count"])

        return self.text, self.compliment_count


    def data_preprocess(self):
        text_slice = self.text[750000:]
        compliment_count_slice = self.compliment_count[750000:]

        tokens  = Tokenizer(filters='', lower=False)
        tokens.fit_on_texts(text_slice)
        text_sequence = tokens.texts_to_sequences(text_slice)

        self.mas_len = len(max(text_sequence, key=len))
        self.vocab_size = len(tokens.word_index)+1

        padded_input= pad_sequences(text_sequence, maxlen=self.mas_len, padding='post')

                    #  it is too large dataset 
                    # categorices are not balanced among all dataset
        X = padded_input
        Y = to_categorical(compliment_count_slice)
        self.y_unique_count = Counter(compliment_count_slice)  # it will count unique number from the list , so that we can pass in the last layer 

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(X, Y, test_size=0.2, random_state=32)

        return self.X_train, self.X_val, self.Y_train, self.Y_val, self.y_unique_count, self.mas_len, self.vocab_size
    

    def neural_models(self):
        print('\n', '--'*10, 'RNN Model', '--'*10 ,'\n')
        rnn_model = Sequential([
        Embedding(self.vocab_size, 64, input_length=self.mas_len),
        SimpleRNN(128, return_sequences=False),
        Dense(len(self.y_unique_count), activation='softmax') 
        ])
        rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
        self.rnn_history = rnn_model.fit(self.X_train, self.Y_train, batch_size=64, validation_data=(self.X_val, self.Y_val), epochs=3)



        print('\n', '--'*10, 'LSTM Model', '--'*10 ,'\n')
        lstm_model = Sequential([
            Embedding(self.vocab_size, 64, input_length=self.mas_len),
            LSTM(128, return_sequences=False),
            Dense(len(self.y_unique_count), activation='softmax') 
            ])
        lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
        self.lstm_history = lstm_model.fit(self.X_train, self.Y_train, batch_size=64, validation_data=(self.X_val, self.Y_val), epochs=3)



        print('\n', '--'*10, 'GRU Model', '--'*10 ,'\n')
        gru_model = Sequential([
            Embedding(self.vocab_size, 64, input_length=self.mas_len),
            GRU(128, return_sequences=False),
            Dense(len(self.y_unique_count), activation='softmax') 
            ])
        gru_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
        self.gru_history = gru_model.fit(self.X_train, self.Y_train, batch_size=64, validation_data=(self.X_val, self.Y_val), epochs=3)

        return self.rnn_history.history['val_accuracy'][-1], self.lstm_history.history['val_accuracy'][-1], self.gru_history.history['val_accuracy'][-1]





path = kagglehub.dataset_download("yelp-dataset/yelp-dataset")

data = []
        #  selecting review one from available ones 
        # make sure it should be "yelp_academic_dataset_review.json" dataset , bcz by using this name causing memory crash everytime  
with open(f'{path}/{os.listdir(path)[4]}', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from line: {line.strip()}\nError: {e}")





obj1 = models()
obj1.data_cleaning()
obj1.data_preprocess()
rnn_val_acc, lstm_val_acc, gru_val_acc = obj1.neural_models()


print("Validation accuracy of RNN model is are : ",rnn_val_acc, '\n' )
print("Validation accuracy of LSTM model is are : ",lstm_val_acc, '\n' )
print("Validation accuracy of GRU model is are : ",gru_val_acc, '\n' )
