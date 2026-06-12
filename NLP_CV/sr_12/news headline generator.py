import kagglehub
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, SimpleRNN, LSTM, GRU, Input
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout


def data_preprocessing(df):
    text_sentence = [str(i).replace('\n', '') for i in df['article']]
    tokens_text = Tokenizer(filters='', lower=False, num_words=13000, oov_token="<unk>")
    tokens_text.fit_on_texts(text_sentence)
    text_sequence_article = tokens_text.texts_to_sequences(text_sentence)


    summary_sentence = [str(s).replace('\n', ' ') + '<end>' for s in df['summary']]
    tokens_summary = Tokenizer(filters='', lower=False, num_words=10000, oov_token="<unk>")
    tokens_summary.fit_on_texts(summary_sentence)
    text_sequence_summary = tokens_summary.texts_to_sequences(summary_sentence)

    mas_len_article = max([len(i) for i in text_sequence_article])
    mas_len_summary = max([len(i) for i in text_sequence_summary])

    dec_input_seq = []
    dec_target_seq = []
    for seq in text_sequence_summary:
        dec_input_seq.append([0]+ seq[:-1])
        dec_target_seq.append(seq)

    end_token_idx = tokens_summary.word_index['<end>']

    x = pad_sequences(text_sequence_article, maxlen=mas_len_article, padding='post')
    dec_input_pad = pad_sequences(dec_input_seq, maxlen=mas_len_summary, padding='post')
    dec_target_pad = pad_sequences(dec_target_seq, maxlen=mas_len_summary, padding='post')

    article_voacb_size = len(tokens_text.word_index) + 1
    summary_vocab_size = len(tokens_summary.word_index) + 1

    # sparse_categorical_crossentropy needs targets with trailing dim
    dec_target_pad = dec_target_pad[..., np.newaxis]

    X_tra, X_val, dec_in_tra, dec_in_val, y_tra, y_val = train_test_split(
            x, dec_input_pad, dec_target_pad, test_size=0.2, random_state=44
        )
    return X_tra, X_val, dec_in_tra, dec_in_val, y_tra, y_val, article_voacb_size, summary_vocab_size, end_token_idx, mas_len_article, mas_len_summary, tokens_text, tokens_summary


HIDDEN = 256
EMB_DIM = 64
DROPOUT= 0.2   

def build_model(mas_len_article, mas_len_summary, article_vocab_size, summary_vocab_size):
    # --- Encoder ---
    enc_input     = Input(shape=(mas_len_article,), name='enc_input')
    enc_embedding = Embedding(article_vocab_size, EMB_DIM, name='enc_emb')(enc_input)
    enc_embedding = Dropout(DROPOUT)(enc_embedding)          # dropout on embeddings
    enc_lstm      = LSTM(HIDDEN, return_state=True, return_sequences=True,
                         dropout=DROPOUT,                    # input dropout
                         recurrent_dropout=0.1,              # keep recurrent low
                         name='enc_lstm')
    enc_out, enc_h, enc_c = enc_lstm(enc_embedding)

    # --- Decoder ---
    dec_input_layer = Input(shape=(mas_len_summary,), name='dec_input')
    dec_embedding   = Embedding(summary_vocab_size, EMB_DIM, name='dec_emb')(dec_input_layer)
    dec_embedding   = Dropout(DROPOUT)(dec_embedding)        # dropout on embeddings
    dec_lstm        = LSTM(HIDDEN, return_sequences=True, return_state=True,
                           dropout=DROPOUT, recurrent_dropout=0.1, name='dec_lstm')
    dec_out, _, _   = dec_lstm(dec_embedding, initial_state=[enc_h, enc_c])
    dec_out         = Dropout(DROPOUT)(dec_out)              # dropout before Dense
    dec_dense       = Dense(summary_vocab_size, activation='softmax', name='dec_dense')
    dec_output      = dec_dense(dec_out)

    training_model = Model([enc_input, dec_input_layer], dec_output)

    # --- Encoder inference model ---
    encoder_model = Model(enc_input, [enc_out, enc_h, enc_c])

    # --- Decoder inference model ---
    inf_dec_input = Input(shape=(1,),      name='inf_dec_input')
    inf_state_h   = Input(shape=(HIDDEN,), name='inf_state_h')
    inf_state_c   = Input(shape=(HIDDEN,), name='inf_state_c')

    inf_dec_emb               = training_model.get_layer('dec_emb')(inf_dec_input)
    inf_dec_out, inf_h, inf_c = training_model.get_layer('dec_lstm')(
        inf_dec_emb, initial_state=[inf_state_h, inf_state_c]
    )
    inf_dec_out = training_model.get_layer('dec_dense')(inf_dec_out)

    decoder_model = Model(
        [inf_dec_input, inf_state_h, inf_state_c],
        [inf_dec_out, inf_h, inf_c]
    )

    return training_model, encoder_model, decoder_model



def model_training(training_model, X_tra, dec_in_tra, y_tra, X_val, dec_in_val, y_val):
    training_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

    training_model.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-4,
            verbose=1
        )
    ]


    history = training_model.fit(
        [X_tra, dec_in_tra],
        y_tra,
        validation_data=([X_val, dec_in_val], y_val),
        epochs=10,
        batch_size=128,
        callbacks=callbacks
    )

    return history




def sample_with_temperature(predictions, temperature=0.8):
    predictions = np.array(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    predictions = np.exp(predictions)
    predictions = predictions / predictions.sum()
    return np.random.choice(len(predictions), p=predictions)


def greedy_search__headline(english_sentence, temperature=0.8):
    seq = tokens_text.texts_to_sequences([english_sentence])
    seq = pad_sequences(seq, maxlen=mas_len_article, padding='post')
    encoder_states, h, c = encoder_model.predict(seq, verbose=0)

    target_token = np.array([[0]])   # <start>
    result = []

    for _ in range(mas_len_summary):
        output, h, c = decoder_model.predict(
            [target_token, h, c, encoder_states], verbose=0
        )
        pred_idx = sample_with_temperature(output[0, 0], temperature)

        # Stop conditions
        if pred_idx == 0:
            break                           # predicted padding — stop
        if pred_idx == end_token_idx:
            break                           # predicted <end> — natural stop
        if pred_idx not in tokens_summary.index_word:
            break

        result.append(tokens_summary.index_word[pred_idx])
        target_token = np.array([[pred_idx]])

    return ' '.join(result)



def beam_search_headline(english_sentence, beam_width=3):
    seq = tokens_text.texts_to_sequences([english_sentence])
    seq = pad_sequences(seq, maxlen=mas_len_article, padding='post')
    h, c = encoder_model.predict(seq, verbose=0)

    beams = [(0.0, [], h, c)]
    completed = []

    for _ in range(mas_len_summary):
        candidates = []
        for log_prob, toks, h, c in beams:
            last = np.array([[toks[-1] if toks else 0]])
            output, new_h, new_c = decoder_model.predict([last, h, c], verbose=0)
            probs = output[0, 0]
            for idx in np.argsort(probs)[-beam_width:]:
                candidates.append((log_prob + np.log(probs[idx] + 1e-8),
                                   toks + [int(idx)], new_h, new_c))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = []
        for log_prob, toks, h, c in candidates[:beam_width]:
            if toks and toks[-1] == end_token_idx:
                completed.append((log_prob, toks))
            else:
                beams.append((log_prob, toks, h, c))
        if not beams:
            break

    if not completed:
        completed = [(b[0], b[1]) for b in beams]

    best = max(completed, key=lambda x: x[0])[1]
    return ' '.join(tokens_summary.index_word[i] for i in best
                    if i in tokens_summary.index_word and i != end_token_idx)







path = kagglehub.dataset_download("arngowda/gigaword-corpus")

df = pd.read_csv(f'{path}/{os.listdir(path)[0]}')
df   = df.sample(150000, random_state=42).reset_index(drop=True)


X_tra, X_val, dec_in_tra, dec_in_val, y_tra, y_val, article_voacb_size, summary_vocab_size, end_token_idx, mas_len_article, mas_len_summary, tokens_text, tokens_summary = data_preprocessing(df)
training_model, encoder_model, decoder_model =build_model(mas_len_article, mas_len_summary, article_voacb_size, summary_vocab_size)
history = model_training(training_model, X_tra, dec_in_tra, y_tra, X_val, dec_in_val, y_val)




sentence = input('Enter article here : ')
print('\n', '--'*10, 'Greedy search headline is :', '--'*10, '\n', greedy_search__headline(sentence))
print('\n', '--'*10, 'Beam search headline is :', '--'*10, '\n', beam_search_headline(sentence))



