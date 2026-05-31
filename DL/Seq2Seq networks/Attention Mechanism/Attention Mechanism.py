import kagglehub
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, LSTM, Input, Concatenate
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout


def load_data():
    path = kagglehub.dataset_download("zainuddin123/parallel-corpus-for-english-urdu-language")
    dir_path = os.listdir(f'{path}')[0]
    files    = os.listdir(f'{path}/{dir_path}')
    eng_file  = [f for f in files if 'eng' in f.lower()][0]
    urdu_file = [f for f in files if 'urd' in f.lower()][0]

    with open(f'{path}/{dir_path}/{eng_file}',  'r', encoding='utf-8') as f:
        eng_lines  = [l.strip() for l in f.readlines()]
    with open(f'{path}/{dir_path}/{urdu_file}', 'r', encoding='utf-8') as f:
        urdu_lines = [l.strip() for l in f.readlines()]

    assert len(eng_lines) == len(urdu_lines), \
        f"Mismatch: {len(eng_lines)} English vs {len(urdu_lines)} Urdu"
    return eng_lines, urdu_lines



def data_preprocess(eng_lines, urdu_lines):
    eng_tokens = Tokenizer(filters='', lower=False)
    eng_tokens.fit_on_texts(eng_lines)
    eng_text_seq = eng_tokens.texts_to_sequences(eng_lines)

    urdu_sentences = [str(s) + ' <end>' for s in urdu_lines]
    urdu_tokens = Tokenizer(filters='')
    urdu_tokens.fit_on_texts(urdu_sentences)
    urdu_text_seq = urdu_tokens.texts_to_sequences(urdu_sentences)

    eng_max_len  = max(len(x) for x in eng_text_seq)
    urdu_max_len = max(len(x) for x in urdu_text_seq)
    end_token_idx = urdu_tokens.word_index['<end>']

    dec_input_seqs, dec_target_seqs = [], []
    for seq in urdu_text_seq:
        dec_input_seqs.append([0] + seq[:-1])
        dec_target_seqs.append(seq)

    X          = pad_sequences(eng_text_seq,    maxlen=eng_max_len,  padding='post')
    dec_input  = pad_sequences(dec_input_seqs,  maxlen=urdu_max_len, padding='post')
    dec_target = pad_sequences(dec_target_seqs, maxlen=urdu_max_len, padding='post')

    eng_vocab_size  = len(eng_tokens.word_index)  + 1
    urdu_vocab_size = len(urdu_tokens.word_index) + 1

    X_tra, X_val, dec_in_tra, dec_in_val, y_tra, y_val = train_test_split(
        X, dec_input, dec_target, test_size=0.2, random_state=32
    )

    return (eng_tokens, urdu_tokens,
            eng_max_len, urdu_max_len,
            eng_vocab_size, urdu_vocab_size,
            end_token_idx,
            X_tra, X_val,
            dec_in_tra, dec_in_val,
            y_tra, y_val)



# MODEL — Bahdanau (additive) attention
#
# WHAT CHANGED vs vanilla:
#   Encoder: return_sequences=True  →  produces ALL hidden states (T_enc, HIDDEN)
#            not just the last one.
#   Attention: a tiny Dense(HIDDEN, tanh) + Dense(1) scores every encoder step
#              against the current decoder step, then softmax gives α weights.
#   Context:  cₜ = Σ αᵢ · hᵢ  (Dot layer with axes=[2,2])
#   Decoder input: concat(dec_embedding, context) fed into LSTM at each step.
#
# WHAT STAYED THE SAME:
#   Tokenisation, teacher-forcing setup, loss, callbacks, translate() logic.
HIDDEN  = 256
EMB_DIM = 128
DROPOUT = 0.3


def build_model_with_attention(eng_max_len, urdu_max_len,
                                eng_vocab_size, urdu_vocab_size):

    # ── ENCODER 
    enc_input     = Input(shape=(eng_max_len,), name='enc_input')
    enc_embedding = Embedding(eng_vocab_size, EMB_DIM, name='enc_emb')(enc_input)
    enc_embedding = Dropout(DROPOUT)(enc_embedding)

    # return_sequences=True → shape (batch, T_enc, HIDDEN)
    # Vanilla only kept the final state; now we keep every step.
    #  here return sequence set tto true , to get outpu at every timestep
    enc_lstm = LSTM(HIDDEN,return_sequences=True, return_state=True, dropout=DROPOUT, recurrent_dropout=0.1, name='enc_lstm')
    enc_out, enc_h, enc_c = enc_lstm(enc_embedding)
    # enc_out : (batch, T_enc, HIDDEN)  — all encoder hidden states
    # enc_h/c : (batch, HIDDEN)         — final state (still used to init decoder)

    # ── DECODER ──────────────────────────────────────────────────────────────
    dec_input_layer = Input(shape=(urdu_max_len,), name='dec_input')
    dec_embedding   = Embedding(urdu_vocab_size, EMB_DIM, name='dec_emb')(dec_input_layer)
    dec_embedding   = Dropout(DROPOUT)(dec_embedding)

    dec_lstm = LSTM(HIDDEN,return_sequences=True,return_state=True,dropout=DROPOUT,recurrent_dropout=0.1,name='dec_lstm')
    dec_out, _, _ = dec_lstm(dec_embedding, initial_state=[enc_h, enc_c])
    # dec_out : (batch, T_dec, HIDDEN)

    # ── ATTENTION (Bahdanau) ─────────────────────────────────────────────────
    # Step 1: score(sₜ, hᵢ) = V · tanh(W1·sₜ + W2·hᵢ)
    #
    # We project encoder and decoder states into the same space, add them,
    # apply tanh, then project to a scalar score per encoder step.

    # W2·hᵢ : project every encoder step
    # shape: (batch, T_enc, HIDDEN)
    enc_proj = Dense(HIDDEN, use_bias=False, name='attn_enc_proj')(enc_out)

    # W1·sₜ : project every decoder step
    # shape: (batch, T_dec, HIDDEN)
    dec_proj = Dense(HIDDEN, use_bias=False, name='attn_dec_proj')(dec_out)

    # KEY CHANGE 2: add projections. We expand dims so broadcasting works:
    #   enc_proj: (batch, 1,     T_enc, HIDDEN)  after expand_dims
    #   dec_proj: (batch, T_dec, 1,     HIDDEN)  after expand_dims
    # Adding gives (batch, T_dec, T_enc, HIDDEN) — every (decoder t, encoder i) pair.
    enc_proj_exp = tf.expand_dims(enc_proj, 1)   # (batch, 1,     T_enc, HIDDEN)
    dec_proj_exp = tf.expand_dims(dec_proj, 2)   # (batch, T_dec, 1,     HIDDEN)
    combined     = tf.nn.tanh(enc_proj_exp + dec_proj_exp)  # (batch, T_dec, T_enc, HIDDEN)

    # V · combined → raw scores, shape: (batch, T_dec, T_enc, 1)
    scores = Dense(1, use_bias=False, name='attn_score')(combined)
    scores = tf.squeeze(scores, axis=-1)          # (batch, T_dec, T_enc)

    # Step 2: softmax over encoder axis → attention weights α
    # shape: (batch, T_dec, T_enc)
    attn_weights = tf.nn.softmax(scores, axis=-1)

    # KEY CHANGE 3: context vector cₜ = Σ αᵢ · hᵢ
    # attn_weights: (batch, T_dec, T_enc)
    # enc_out     : (batch, T_enc, HIDDEN)
    # matmul gives: (batch, T_dec, HIDDEN) — one context vector per decoder step
    context = tf.matmul(attn_weights, enc_out)    # (batch, T_dec, HIDDEN)

    # KEY CHANGE 4: concat context with decoder output, then project to vocab
    dec_combined = Concatenate(axis=-1, name='dec_context_concat')([dec_out, context])
    dec_combined = Dropout(DROPOUT)(dec_combined)
    dec_output   = Dense(urdu_vocab_size, activation='softmax', name='dec_dense')(dec_combined)

    # ── TRAINING MODEL ───────────────────────────────────────────────────────
    training_model = Model([enc_input, dec_input_layer], dec_output)

    # ── ENCODER INFERENCE MODEL (same as before) ─────────────────────────────
    # Returns all hidden states + final state for attention lookup at decode time
    encoder_model = Model(enc_input, [enc_out, enc_h, enc_c])

    # ── DECODER INFERENCE MODEL ───────────────────────────────────────────────
    # At inference we decode ONE token at a time.
    # Input:  single token + previous LSTM state + ALL encoder states (for attention)
    # Output: probability distribution + new LSTM state

    inf_dec_input = Input(shape=(1,),       name='inf_dec_input')
    inf_state_h   = Input(shape=(HIDDEN,),  name='inf_state_h')
    inf_state_c   = Input(shape=(HIDDEN,),  name='inf_state_c')
    inf_enc_out   = Input(shape=(eng_max_len, HIDDEN), name='inf_enc_out')  # all enc states

    # embed the single token
    inf_emb = training_model.get_layer('dec_emb')(inf_dec_input)

    # run decoder LSTM for one step
    inf_dec_out, inf_h, inf_c = training_model.get_layer('dec_lstm')(
        inf_emb, initial_state=[inf_state_h, inf_state_c]
    )                                             # inf_dec_out: (batch, 1, HIDDEN)

    # compute attention for this single decoder step
    inf_enc_proj = training_model.get_layer('attn_enc_proj')(inf_enc_out)
    inf_dec_proj = training_model.get_layer('attn_dec_proj')(inf_dec_out)

    inf_enc_exp  = tf.expand_dims(inf_enc_proj, 1)
    inf_dec_exp  = tf.expand_dims(inf_dec_proj, 2)
    inf_combined = tf.nn.tanh(inf_enc_exp + inf_dec_exp)

    inf_scores      = training_model.get_layer('attn_score')(inf_combined)
    inf_scores      = tf.squeeze(inf_scores, axis=-1)
    inf_attn_weights = tf.nn.softmax(inf_scores, axis=-1)

    inf_context  = tf.matmul(inf_attn_weights, inf_enc_out)   # (batch, 1, HIDDEN)

    # concat + dense — reuse trained weights
    inf_dec_combined = Concatenate(axis=-1, name='inf_concat')([inf_dec_out, inf_context])
    inf_output       = training_model.get_layer('dec_dense')(inf_dec_combined)

    decoder_model = Model(
        [inf_dec_input, inf_state_h, inf_state_c, inf_enc_out],
        [inf_output, inf_h, inf_c]
    )

    return training_model, encoder_model, decoder_model




eng_lines, urdu_lines = load_data()

(eng_tokens, urdu_tokens,
 eng_max_len, urdu_max_len,
 eng_vocab_size, urdu_vocab_size,
 end_token_idx,
 X_tra, X_val,
 dec_in_tra, dec_in_val,
 y_tra, y_val) = data_preprocess(eng_lines, urdu_lines)

training_model, encoder_model, decoder_model = build_model_with_attention(
    eng_max_len, urdu_max_len, eng_vocab_size, urdu_vocab_size
)

training_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

training_model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=2, min_lr=1e-5, verbose=1)
]

history = training_model.fit(
    [X_tra, dec_in_tra], y_tra,
    validation_data=([X_val, dec_in_val], y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks
)



# INFERENCE
# KEY CHANGE in translate(): encoder_model now returns enc_out (all states).
# decoder_model now takes enc_out as an extra input for attention.
def sample_with_temperature(predictions, temperature=0.8):
    predictions = np.array(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    predictions = np.exp(predictions)
    predictions /= predictions.sum()
    return np.random.choice(len(predictions), p=predictions)


def translate(english_sentence, temperature=0.8):
    seq = eng_tokens.texts_to_sequences([english_sentence])
    seq = pad_sequences(seq, maxlen=eng_max_len, padding='post')

    # encoder now gives us ALL hidden states + final state
    enc_out, h, c = encoder_model.predict(seq, verbose=0)
    # enc_out: (1, eng_max_len, HIDDEN)

    target_token = np.array([[0]])    # <start>
    result = []

    for _ in range(urdu_max_len):
        # KEY CHANGE: pass enc_out to decoder at every step
        output, h, c = decoder_model.predict(
            [target_token, h, c, enc_out], verbose=0
        )
        pred_idx = sample_with_temperature(output[0, 0], temperature)

        if pred_idx == 0:               break
        if pred_idx == end_token_idx:   break
        if pred_idx not in urdu_tokens.index_word: break

        result.append(urdu_tokens.index_word[pred_idx])
        target_token = np.array([[pred_idx]])

    return ' '.join(result)


sentence = input("Enter English sentence: ")
print("Translation:", translate(sentence))











            #  checking bleu score of the model 

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def evaluate_bleu(num_samples=200):
    references, hypotheses = [], []
    indices = np.random.choice(len(X_val), num_samples, replace=False)

    for idx in indices:
        ref_ids   = y_val[idx]
        ref_words = [urdu_tokens.index_word[i]
                     for i in ref_ids if i != 0 and i != end_token_idx]

        eng_seq = X_val[idx:idx+1]
        enc_out, h, c = encoder_model.predict(eng_seq, verbose=0)

        target_token    = np.array([[0]])
        predicted_words = []

        for _ in range(urdu_max_len):
            output, h, c = decoder_model.predict(
                [target_token, h, c, enc_out], verbose=0   # enc_out added
            )
            pred_idx = np.argmax(output[0, 0])

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