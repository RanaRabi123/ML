import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from category_encoders import CountEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf

df = pd.read_csv('/content/owid-energy-data.csv')

df.isnull().sum()

ce_encode = CountEncoder()
df['country'] = ce_encode.fit_transform(df['country'])

            # to drop columns that have more than 60% empty data
for  i in df.columns:
  if (df[i].isnull().sum()/len(df) < 0.6) & (df[i].isnull().sum()/len(df) > 0.0) :
    print(df[i].isnull().sum()/len(df))
    df.drop(i, axis=1, inplace=True)


            #  filling empty data with median 
for i in df.columns:
  if df[i].isnull().sum() != 0:
    imputer = SimpleImputer(strategy='median')
    df[i] = imputer.fit_transform(df[i].values.reshape(-1, 1))


            #  if imputer did not work for first and last data point   
df = df.fillna(df.median(numeric_only=True))


x = df.drop('renewables_share_elec', axis=1)
y = df['renewables_share_elec']

            #  selecting best features from 100+
selector = SelectKBest(score_func=f_classif, k=30)  # only select top fetaure
x_selected = selector.fit_transform(x, y)

selected_feature_names = x.columns[selector.get_support()]  # names of the features selected by SelectKBest

x_selected_df = pd.DataFrame(x_selected, columns=selected_feature_names)    # Convert the np array X_selected into pd DataFrame

# Add 'year' and 'country' columns from the original df to this new DataFrame
x_selected_df['year'] = x['year'].copy()
x_selected_df['country'] = x['country'].copy()

# Reassign X_selected to this new DataFrame to be used in subsequent steps
x_selected = x_selected_df


x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.2, random_state=42)


exclude = ['country', 'year']   # columns don't need to scaled
scale_cols = [c for c in x_train.columns if c not in exclude]

scaler = StandardScaler()
x_train[scale_cols] = scaler.fit_transform(x_train[scale_cols])
x_test[scale_cols] = scaler.transform(x_test[scale_cols])


lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)


metrics = {
    'mse': mean_squared_error(y_test, y_pred),
    'r2_score': r2_score(y_test, y_pred),
    'mae': mean_absolute_error(y_test, y_pred)
}

print(metrics)













                # when doing manually 
                #  not working yet 
xx = tf.constant(x_train, dtype=tf.float32)
yy = tf.constant(y_train, dtype=tf.float32)


# w = tf.Variable(0.0)
w = tf.Variable(tf.zeros(xx.shape[1], dtype=tf.float32))
b = tf.Variable(0.0)


learning_rate = 0.0001

for epoch in range(100):

    with tf.GradientTape() as tape:
        y_pred = tf.matmul(xx, tf.reshape(w, (-1, 1))) + b
        loss = tf.reduce_mean((yy - y_pred)**2)

    dw, db = tape.gradient(loss, [w, b])

    w.assign_sub(learning_rate * dw)
    b.assign_sub(learning_rate * db)

    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}, w: {w.numpy()}, b: {b.numpy()}")