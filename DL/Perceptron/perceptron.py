import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

from mlxtend.plotting import plot_decision_regions

df = pd.read_excel('/content/Titanic.xlsx')

df.drop(['number', 'sibsp', 'ticket', 'parch', 'cabin', 'embarked', 'boat', 'body', 'home.dest'], axis=1, inplace=True)
df.dropna(axis=0, inplace=True)

df = df[df['fare'] <= 277]      # removing outlier fare , so that visualize can be clear

encoder = LabelEncoder()
encoder.fit(df['sex'])
df['sex'] = encoder.transform(df['sex'])

x = df.drop(['survived', 'sex', 'pclass'], axis=1)
y = df['survived']  

p = Perceptron()
p.fit(x, y)

print('Weight W value in this case is ', p.coef_)
print('Bias b value in this case is : ', p.intercept_)

print('This graph is just fro visualization ')
plot_decision_regions(x.to_numpy() , y.to_numpy() , clf=p, legend=2)