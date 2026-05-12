import kagglehub
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, SimpleRNN, LSTM, GRU, Input
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout


def load_data():
    path = kagglehub.dataset_download("zainuddin123/parallel-corpus-for-english-urdu-language")
    dir_path= os.listdir(f'{path}')[0]

    files = os.listdir(f'{path}/{dir_path}')        
    eng_file  = [f for f in files if 'eng' in f.lower()][0]     #sometimes it give english file first and someitmes urdu first  
    urdu_file = [f for f in files if 'urd' in f.lower()][0]

    df_eng = f'{path}/{dir_path}/{eng_file}'
    df_urdu = f'{path}/{dir_path}/{urdu_file}'

    with open(df_eng, 'r', encoding='utf-8') as file:
        eng_liness = file.readlines()

    with open(df_urdu, 'r', encoding='utf-8') as file:
        urdu_liness = file.readlines()

    eng_lines = [line.strip() for line in eng_liness]
    urdu_lines = [line.strip() for line in urdu_liness]


        # it will thorw the error if lenght is not same 
    assert len(eng_lines) == len(urdu_lines), \
    f"Mismatch: {len(eng_lines)} English vs {len(urdu_lines)} Urdu"

    return eng_lines, urdu_lines



def data_preprocess():
    eng_tokens = Tokenizer(filters='', lower=False)
    eng_tokens.fit_on_texts(eng_lines)
    eng_text_seq = eng_tokens.texts_to_sequences(eng_lines)

    # Add <end> to every Urdu sentence BEFORE fitting the tokenizer
    # so <end> gets its own index
    urdu_sentences = [str(s) + ' <end>' for s in urdu_lines]
    urdu_tokens = Tokenizer(filters='')
    urdu_tokens.fit_on_texts(urdu_sentences)
    urdu_text_seq = urdu_tokens.texts_to_sequences(urdu_sentences)

    eng_max_len  = max(len(x) for x in eng_text_seq)
    urdu_max_len = max(len(x) for x in urdu_text_seq)

    end_token_idx = urdu_tokens.word_index['<end>']   # save this

    # Teacher forcing setup
    # decoder_input : <start>(0), w1, w2, ...
    # decoder_target: w1, w2, ..., <end>
    dec_input_seqs  = []
    dec_target_seqs = []
    for seq in urdu_text_seq:
        dec_input_seqs.append([0] + seq[:-1])
        dec_target_seqs.append(seq)

    X          = pad_sequences(eng_text_seq,   maxlen=eng_max_len,  padding='post')
    dec_input  = pad_sequences(dec_input_seqs,  maxlen=urdu_max_len, padding='post')
    dec_target = pad_sequences(dec_target_seqs, maxlen=urdu_max_len, padding='post')

    eng_vocab_size  = len(eng_tokens.word_index)  + 1
    urdu_vocab_size = len(urdu_tokens.word_index) + 1

    X_tra, X_val, dec_in_tra, dec_in_val, y_tra, y_val = train_test_split(
        X, dec_input, dec_target, test_size=0.2, random_state=32
    )

    return (
        eng_tokens, urdu_tokens,
        eng_max_len, urdu_max_len,
        eng_vocab_size, urdu_vocab_size,
        end_token_idx,
        X_tra, X_val,
        dec_in_tra, dec_in_val,
        y_tra, y_val
    )



eng_lines , urdu_lines = load_data()

(eng_tokens, urdu_tokens,
 eng_max_len, urdu_max_len,
 eng_vocab_size, urdu_vocab_size,
 end_token_idx,
 X_tra, X_val,
 dec_in_tra, dec_in_val,
 y_tra, y_val) = data_preprocess()




HIDDEN  = 256
EMB_DIM = 128
DROPOUT = 0.3   # tune this: try 0.2–0.4

def build_model(eng_max_len, urdu_max_len, eng_vocab_size, urdu_vocab_size):
    # --- Encoder ---
    enc_input     = Input(shape=(eng_max_len,), name='enc_input')
    enc_embedding = Embedding(eng_vocab_size, EMB_DIM, name='enc_emb')(enc_input)
    enc_embedding = Dropout(DROPOUT)(enc_embedding)          # dropout on embeddings
    enc_lstm      = LSTM(HIDDEN, return_state=True, return_sequences=True,
                         dropout=DROPOUT,                    # input dropout
                         recurrent_dropout=0.1,              # keep recurrent low
                         name='enc_lstm')
    enc_out, enc_h, enc_c = enc_lstm(enc_embedding)

    # --- Decoder ---
    dec_input_layer = Input(shape=(urdu_max_len,), name='dec_input')
    dec_embedding   = Embedding(urdu_vocab_size, EMB_DIM, name='dec_emb')(dec_input_layer)
    dec_embedding   = Dropout(DROPOUT)(dec_embedding)        # dropout on embeddings
    dec_lstm        = LSTM(HIDDEN, return_sequences=True, return_state=True,
                           dropout=DROPOUT, recurrent_dropout=0.1, name='dec_lstm')
    dec_out, _, _   = dec_lstm(dec_embedding, initial_state=[enc_h, enc_c])
    dec_out         = Dropout(DROPOUT)(dec_out)              # dropout before Dense
    dec_dense       = Dense(urdu_vocab_size, activation='softmax', name='dec_dense')
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

training_model, encoder_model, decoder_model = build_model(
    eng_max_len, urdu_max_len, eng_vocab_size, urdu_vocab_size
)

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
        min_lr=1e-5,
        verbose=1
    )
]

history = training_model.fit(
    [X_tra, dec_in_tra],
    y_tra,
    validation_data=([X_val, dec_in_val], y_val),
    epochs=50,              
    batch_size=64,
    callbacks=callbacks
)



def sample_with_temperature(predictions, temperature=0.8):
    predictions = np.array(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    predictions = np.exp(predictions)
    predictions = predictions / predictions.sum()
    return np.random.choice(len(predictions), p=predictions)



def translate(english_sentence, temperature=0.8):
    seq = eng_tokens.texts_to_sequences([english_sentence])
    seq = pad_sequences(seq, maxlen=eng_max_len, padding='post')
    _, h, c = encoder_model.predict(seq, verbose=0)

    target_token = np.array([[0]])   # <start>
    result = []

    for _ in range(urdu_max_len):
        output, h, c = decoder_model.predict(
            [target_token, h, c], verbose=0
        )
        pred_idx = sample_with_temperature(output[0, 0], temperature)

        # Stop conditions
        if pred_idx == 0:
            break                           # predicted padding — stop
        if pred_idx == end_token_idx:
            break                           # predicted <end> — natural stop 
        if pred_idx not in urdu_tokens.index_word:
            break

        result.append(urdu_tokens.index_word[pred_idx])
        target_token = np.array([[pred_idx]])

    return ' '.join(result)


sentence = input("Enter English sentence: ")
print("Translation:", translate(sentence))









#  checking the accuracy if the model using BLEU , which is standard metric for checking translation result. 


from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def evaluate_bleu(num_samples=200):
    references = []
    hypotheses = []

    # Sample from validation set
    indices = np.random.choice(len(X_val), num_samples, replace=False)

    for idx in indices:
        # Get reference translation
        ref_ids = y_val[idx]
        ref_words = [
            urdu_tokens.index_word[i]
            for i in ref_ids
            if i != 0 and i != end_token_idx
        ]

        # Get model translation
        eng_seq = X_val[idx:idx+1]
        _, h, c = encoder_model.predict(eng_seq, verbose=0)

        target_token = np.array([[0]])
        predicted_words = []

        for _ in range(urdu_max_len):
            output, h, c = decoder_model.predict(
                [target_token, h, c], verbose=0
            )
            pred_idx = np.argmax(output[0, 0])   # greedy for eval

            if pred_idx == 0 or pred_idx == end_token_idx:
                break

            if pred_idx in urdu_tokens.index_word:
                predicted_words.append(urdu_tokens.index_word[pred_idx])

            target_token = np.array([[pred_idx]])

        references.append([ref_words])
        hypotheses.append(predicted_words)

    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    print(f"BLEU score: {bleu:.4f}  ({bleu*100:.1f}/100)")
    return bleu

evaluate_bleu()