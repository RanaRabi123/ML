import regex as re
import tensorflow_datasets as tfds
from nltk import word_tokenize 
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class text_cleaning(BaseEstimator, TransformerMixin):
    def clean_text(self, text):
            # so that it cann handle both , string and byte type
        text = text.decode('utf-8') if isinstance(text, bytes) else text
        text = re.sub(r"<[^>]+>", " ", text)                      # remove HTML
        text = re.sub(r"\s+", " ", text).strip()                  # normalize spaces
        text = text.replace("\'", "'")                            # fix escaped quotes
        return text

    def transform(self, x):
        x_cleaned = x.apply(self.clean_text)
        return x_cleaned.apply(word_tokenize)
        
        
    def fit(self, x, y=None):
        return self

(ds_train, ds_test), ds_info = tfds.load(
            'imdb_reviews',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True
        )

df = tfds.as_dataframe(ds_train, ds_info)   # we nned to pass ds_info bcz it has the info of the dataset 
x = df['text']
y = df['label']


pipeline = Pipeline([
    ('cleaner', text_cleaning())
])


cleaning_pipeline = pipeline.fit_transform(x)
