from transformers import AutoTokenizer, AutoModelForSequenceClassification
import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments
from datasets import Dataset
from transformers import Trainer
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import time
import regex as re
import emoji
import html



# def tokenize_function(examples, tokenizer):
#     return tokenizer(examples, padding="max_length", truncation=True)

def data_processing(tweet, class_label, name):
    tokenizer = AutoTokenizer.from_pretrained(name)

    encoded = []
    for i in tweet:
        encoded.append(tokenizer(i, padding='max_length', truncation=True))

    encoded_token = [encoded[i]['input_ids'] for i in range(len(encoded))]

    X_train, X_test, y_train, y_test = train_test_split(encoded_token, class_label, test_size=0.2, random_state=55)

    return X_train, X_test, y_train, y_test, tokenizer


def model_training(name, X_train, X_test, y_train, y_test, original_class_labels): # Added original_class_labels
    print(('--')*10, f'Initializing Model : {name}' ,('--')*10)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3)

    training_args = TrainingArguments(
        num_train_epochs=3,
        output_dir="./results",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        weight_decay=0.01
    )

    train_dataset = Dataset.from_dict({'input_ids': X_train, 'labels': y_train})
    val_dataset = Dataset.from_dict({'input_ids': X_test, 'labels': y_test})

    # weights = compute_class_weight(
    # class_weight="balanced",
    # classes=np.unique(original_class_labels), # Used original_class_labels
    # y=original_class_labels                    # Used original_class_labels
    # )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
        # class_weights=weights
    )
    trainer.train()

    return trainer, model


def benchmarking(tweet, class_label):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model_names = ['bert-base-cased',
                'distilbert-base-uncased',
                'FacebookAI/roberta-base']

    for name in model_names:
        print(f"=== Benchmarking {name} ===")

        # CALL YOUR FUNCTIONS: Pass the names, and catch the returned assets
        X_train, X_test, y_train, y_test, tokenizer = data_processing(tweet, class_label, name)
        trainer, model = model_training(name, X_train, X_test, y_train, y_test, class_label)  # Passed class_label

        # 2. Extract a sample input from your tokenized split data
        # (We use a slice of size 1, e.g., the first test sample, for single-sequence latency)
        sample_input = {key: tf.convert_to_tensor(val[:1]) for key, val in X_test.items()}

        # 3. Warm-up iterations (Forces TensorFlow graph compilation)
        for _ in range(5):
            _ = model(sample_input, training=False)

        # Reset hardware statistics right before the testing loop
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.reset_memory_stats('GPU:0')

        # 4. Run the isolated performance test loop
        num_runs = 50
        start_time = time.perf_counter()

        for _ in range(num_runs):
            # We pass training=False to turn off dropout layers for a clean test
            _ = model(sample_input, training=False)

        end_time = time.perf_counter()

        # 5. Compute and display metrics
        avg_latency = ((end_time - start_time) * 1000) / num_runs
        print(f"⏱️ Avg Latency: {avg_latency:.2f} ms")

        if gpus:
            peak_mem = tf.config.experimental.get_memory_info('GPU:0')['peak'] / (1024 ** 2)
            print(f"💾 Peak GPU Memory: {peak_mem:.2f} MB")

        # 6. Force clean memory to prevent bleed into the next model iteration
        del model, trainer, X_train, X_test
        tf.keras.backend.clear_session()
        print("-" * 40 + "\n")
        
        return trainer, tokenizer


def user_prediction(trainer, tokenizer):
    user_input = input('Enter any text : ')
    user_encode = tokenizer(user_input, padding="max_length", truncation=True)

    single_example_dataset = Dataset.from_dict({
        'input_ids': [user_encode['input_ids']],
        'token_type_ids': [user_encode['token_type_ids']],
        'attention_mask': [user_encode['attention_mask']]
    })

    prob = trainer.predict(single_example_dataset)
    pred_class = np.argmax(prob[0], axis=1)

    print('The prediction class is : ', pred_class)




path = kagglehub.dataset_download("thedevastator/hate-speech-and-offensive-language-detection")

df = pd.read_csv(f'{path}/train.csv')
df   = df.sample(13000, random_state=42).reset_index(drop=True)


tweet = []
for i in df['tweet']:
  tweet.append(re.sub(r"https?://\S+|www\.\S+|\'s|@\S+|\'s|<[^>]+>|\\|&lt;[^&]*&gt;|;|#\d+;?s?|!*|[\"]|\RT+|\.+", '', emoji.replace_emoji(html.unescape(i), replace="")))

class_label = df['class']





trainer , tokenizer = benchmarking(tweet, class_label)
user_prediction(trainer, tokenizer)