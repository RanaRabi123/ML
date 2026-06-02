import kagglehub
import pandas as pd
import numpy as np
import os 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

class vectorize():
    def __init__(self):
        path = kagglehub.dataset_download("bhavikjikadara/bbc-news-articles")
        self.df = pd.read_csv(f'{path}/{os.listdir(path)[0]}')

    def clean_preprocess(self):
        drop_list = ['pubDate', 'guid', 'link']
        self.df.drop(drop_list, axis=1, inplace=True)

        return self.df
    
    def vectorization(self):
        clean_df = self.clean_preprocess()
        user_choice = int(input("Which one Vectorization you want to select : \n 1. CountVectorizer \n 2. TfidVectorizer"))
        if user_choice==1:
            vectorizer = CountVectorizer(stop_words='english')
            print("\n","--"*10,"Vectorizing with CountVectorizer ", "--"*10, "\n")
            print("\n","--"*10,"Top vectorizer terms/words", "--"*10, "\n")
        elif user_choice==2:
            vectorizer = TfidfVectorizer(stop_words='english')
            print("\n","--"*10, "Vectorizing with TfidVectorizer ","--"*10, "\n")
            print("\n","--"*10,"Top vectorizer terms/words", "--"*10, "\n")
        else:
            print("Invalid choice ... ")

        preprocess = ColumnTransformer([
                ("title", clone(vectorizer), "title"),
                ("desc", clone(vectorizer), "description")
            ])


        final_vector = preprocess.fit_transform(clean_df)

        scores = np.asarray(final_vector.mean(axis=0)).flatten()
        # as we sort from low to high, means we will from the end of array, [-20:] means top 20 , and [::-1] means last one come at first 
        top_indices = scores.argsort()[-20:][::-1] 
        print(preprocess.get_feature_names_out()[top_indices])
        
        return final_vector



obj1 = vectorize()
obj1.vectorization()