import pandas as pd
import numpy as np
import os 
import regex as re
import html
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from collections import Counter
import kagglehub


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim//num_heads
        )

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)

        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)



class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()

        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.pos_emb = layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim
        )

    def call(self, x):
        maxlen = tf.shape(x)[-1]

        positions = tf.range(start=0, limit=maxlen, delta=1)

        positions = self.pos_emb(positions)

        x = self.token_emb(x)

        return x + positions




def model_training():
    embed_dim = 64
    num_heads = 2
    ff_dim = 128

    # Build Model

    inputs = layers.Input(shape=(max_len,))

    embedding_layer = TokenAndPositionEmbedding(
        max_len,
        vocab_size,
        embed_dim
    )

    x = embedding_layer(inputs)

    transformer_block = TransformerBlock(
        embed_dim,
        num_heads,
        ff_dim
    )

    x = transformer_block(x)
            # as we need two transformer layers/blocks
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(label_count, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)


    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, callbacks= [callback] )

    return history, model



def text_prediction():

    test_text = input('enter any text')

    test_seq = tokenizer.texts_to_sequences(test_text)
    test_pad = tf.keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=max_len, padding='post')

    probs      = model.predict(test_pad)[0]   # shape (4,)
    pred_class = np.argmax(probs)             # index of highest probability
    confidence = probs[pred_class]

    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    print(f"It belongs to {class_names[pred_class]} — {confidence*100:.1f}%")








# Download latest version
path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")

df = pd.read_csv(f'{path}/{os.listdir(path)[0]}')
df.head()


description = []
for i in df['Description']:
        # tetx is too messy adn contains many broken tags  
  description.append(re.sub(r"https?://\S+|www\.\S+|\'s|@\S+|\'s|<[^>]+>|\\|&lt;[^&]*&gt;|;|#\d+;?s?", '', html.unescape(i)))

texts = description
labels = df['Class Index']-1
label_count= len(Counter(labels))

tokenizer = tf.keras.preprocessing.text.Tokenizer()
vocab_size = len(tokenizer.word_index)+1
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_len = max(len(i) for i in sequences)

padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen=max_len,padding='post')




history, model = model_training()
text_prediction()
