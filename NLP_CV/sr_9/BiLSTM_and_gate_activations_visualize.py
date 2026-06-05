import pandas as pd
import os
from keras.models import Sequential, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras.layers import Embedding, Dense , LSTM, Bidirectional, Dropout
import matplotlib.pyplot as plt
import kagglehub


def data_process(df):
    drop_col = ['name', 'country', 'date_time', 'review_head']
    for i in drop_col:
        df.drop(i, axis=1, inplace=True)
    df.dropna(inplace=True)


    token = Tokenizer(filters='', lower=False)
    token.fit_on_texts(df['review_body'])
    text_sequence = token.texts_to_sequences(df['review_body'])

    mas_len = max([len(i) for i in text_sequence])
    vocab_size = len(token.word_index) + 1

    padded_input = pad_sequences(text_sequence, maxlen=mas_len, padding='post')


    x = padded_input
    y = to_categorical(df['stars'] - 1, num_classes=5)
    y_unique_count = Counter(df['stars'])   # to get unique_count for last layer neurons 
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=31, test_size=0.2)

    return x_train, x_test, y_train, y_test, y_unique_count, vocab_size, mas_len



def lstm_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=mas_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dense(32, activation='relu'),
        Dense(len(y_unique_count), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    callback = EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64, callbacks=[callback])
    return history, model




def visualization(model, x_test, history):
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


                # LSTM hidden state activations over time 
                # Get output of the first BiLSTM layer (return_sequences=True)
    bilstm_layer = model.layers[1]
    activation_model = Model(inputs=model.layers[0].input, outputs=bilstm_layer.output)

    # activations shape: (batch, timesteps, 128) --> 64(forward)+64(backward)
    activations = activation_model.predict(x_test[:1])  # one sample

    # Plot activations across timesteps for first 10 units
    plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.plot(activations[0, :, i], alpha=0.6, label=f'Unit {i}')
    plt.title('BiLSTM Hidden State Activations Over Time (first 10 units)')
    plt.xlabel('Timestep')
    plt.ylabel('Activation')
    plt.legend(loc='upper right', fontsize=7)
    plt.show()

                # Heatmap of activations (timesteps x units) 
    plt.figure(figsize=(12, 4))
    plt.imshow(activations[0, :50, :64].T, aspect='auto', cmap='RdBu_r')
    plt.colorbar(label='Activation value')
    plt.title('BiLSTM Activation Heatmap (first 50 timesteps, forward units)')
    plt.xlabel('Timestep')
    plt.ylabel('Hidden unit')
    plt.show()






path = kagglehub.dataset_download("yanmaksi/reviews-data-for-classification-model")

df = pd.read_csv(f'{path}/{os.listdir(path)[0]}')
df.head()

x_train, x_test, y_train, y_test, y_unique_count, vocab_size, mas_len= data_process(df)
history, model = lstm_model()
visualization(model, x_test, history)