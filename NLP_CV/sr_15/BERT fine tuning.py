from transformers import AutoTokenizer, AutoModelForSequenceClassification
import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments
from datasets import Dataset
from transformers import Trainer
import tensorflow as tf


def tokenize_function(examples, tokenizer):
    return tokenizer(examples, padding="max_length", truncation=True)

def data_processing(df, tokenizer):
    
    encoded = []
    for i in df['According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .']:
        encoded.append(tokenizer(i, padding='max_length', truncation=True))

    encoded_token = [encoded[i]['input_ids'] for i in range(len(encoded))]

    le = LabelEncoder()
    label_encode = le.fit_transform(df['neutral'])

    X_train, X_test, y_train, y_test = train_test_split(encoded_token, label_encode, test_size=0.2, random_state=55)

    return X_train, X_test, y_train, y_test, le


def model_training(X_train, X_test, y_train, y_test):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01
    )

    train_dataset = Dataset.from_dict({'input_ids': X_train, 'labels': y_train})
    val_dataset = Dataset.from_dict({'input_ids': X_test, 'labels': y_test})

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


path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")

df = pd.read_csv(f'{path}/all-data.csv', encoding='latin-1')
df.dropna(inplace=True, axis=0)

df.head()

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

X_train, X_test, y_train, y_test, le = data_processing(df, tokenizer)
trainer = model_training(X_train, X_test, y_train, y_test)

user_prediction(le)


