import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


import kagglehub

# Download latest version
path = kagglehub.dataset_download("paresh2047/uci-semcom")

import os 
file_name= os.listdir(path)[0]

df = pd.read_csv(f'{path}/{file_name}')
df.head()

df.drop('Time', axis=1, inplace=True)
df.fillna(df.median(), inplace=True)


X = df.drop('Pass/Fail', axis=1)
y = df['Pass/Fail']


selector = SelectKBest(score_func=f_classif, k=50)  # only select top 50 important fetaure
X_selected = selector.fit_transform(X, y)
print(X_selected.shape)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)                       # labels 1, -1 
y_pred_prob= lr_model.predict_proba(X_test)[:, 1]       #probability of  class 1 (Fail)


metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc_score': roc_auc_score(y_test, y_pred_prob),
    'classification_report': classification_report(y_test, y_pred)
}
