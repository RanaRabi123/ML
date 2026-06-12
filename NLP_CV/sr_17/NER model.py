from transformers import pipeline

new_pipeline = pipeline(task='ner', model='dslim/bert-base-NER', tokenizer='dslim/bert-base-NER', aggregation_strategy='simple')

user_input = input('Enter any task : ')
entities= new_pipeline(user_input)

for i in entities:
    print('Entity     : ', i['word'])
    print('Type       : ', i['entity_group'])
    print('Confidence : ', i['score'])
    print('--'*15)






from transformers import AutoTokenizer, AutoModelForSequenceClassification
import kagglehub
import pandas as pd
import numpy as np
import regex as re
import os 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments
from datasets import Dataset
from transformers import Trainer
import tensorflow as tf
from collections import Counter


def tokenize_function(examples, tokenizer):
    return tokenizer(examples, padding="max_length", truncation=True)

def data_processing(df, tokenizer):
    
    encoded = []
    for i in sentence:
        encoded.append(tokenizer(i, padding='max_length', truncation=True))

    # encoded_token = [encoded[i]['input_ids'] for i in range(len(encoded))]

    le = LabelEncoder()
    label_encode = le.fit_transform(label)
    label_count = Counter(label_encode)

    X_train, X_test, y_train, y_test = train_test_split(encoded, label_encode, test_size=0.2, random_state=55)

    return X_train, X_test, y_train, y_test, le, label_count


def model_training(X_train, X_test, y_train, y_test,label_count):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=label_count)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01
    )

    x_train_input_ids = [X_train[i]['input_ids'] for i in range(len(X_train))]
    x_train_token_ids = [X_train[i]['token_type_ids'] for i in range(len(X_train))]
    x_train_attention_mask = [X_train[i]['attention_mask'] for i in range(len(X_train))]

    x_test_input_ids = [X_test[i]['input_ids'] for i in range(len(X_test))]
    x_test_token_ids = [X_test[i]['token_type_ids'] for i in range(len(X_test))]
    x_test_attention_mask = [X_test[i]['attention_mask'] for i in range(len(X_test))]


    train_dataset = Dataset.from_dict({'input_ids': x_train_input_ids,
                                       'token_type_ids': x_train_token_ids,
                                       'attention_mask': x_train_attention_mask,
                                       'labels': y_train})
    val_dataset = Dataset.from_dict({'input_ids': x_test_input_ids,
                                     'token_type_ids': x_test_token_ids,
                                     'attention_mask': x_test_attention_mask,
                                     'labels': y_test})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    return trainer


def user_prediction(le):
    user_input = input('Enter any text : ')
    user_encode = tokenize_function(user_input, tokenizer)

    single_example_dataset = Dataset.from_dict({
        'input_ids': [user_encode['input_ids']],
        'token_type_ids': [user_encode['token_type_ids']],
        'attention_mask': [user_encode['attention_mask']]
    })

    prob = trainer.predict(single_example_dataset)
    pred_class = np.argmax(prob[0], axis=1)
    pred_class_name = le.inverse_transform(pred_class)
    
    return pred_class_name[0]



path = kagglehub.dataset_download("juliangarratt/conll2003-dataset")

file_path = os.path.join(path, os.listdir(path)[0], 'eng.train')

with open(file_path, 'r', encoding='latin-1') as f:
    file = f.readlines()


sentence=[]
label=[]
for i in file:
  if len(i)<3:
    print('ok')
  else:
    label.append(re.sub(r'-|B-', '', i.split()[-1]))
    sentence.append(re.sub(r'-', '', i.split()[0]))


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

X_train, X_test, y_train, y_test, le ,label_count= data_processing(sentence, label, tokenizer)
trainer = model_training(X_train, X_test, y_train, y_test,label_count)

user_prediction(le)


