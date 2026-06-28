import mlflow
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score


def data_cleaning():
    path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
    df = pd.read_csv(f"{path}/WELFake_Dataset.csv")

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.drop(['title', 'Unnamed: 0'], inplace=True, axis=1)
    df['text']=df['text'].str.replace(r'^.*?-','', regex=True)

    return df


def data_preprocess():
    df = data_cleaning()

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = data_preprocess()

lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

svc_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LinearSVC())
])



mlflow.set_experiment("Fake News Detection Experiment")

with mlflow.start_run(run_name="Fake News Detection Logistic Regression Model"):
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)

    print('--'*10, "Logistic Regression Classification Report :")
    print(classification_report(y_test, lr_pred, target_names=['fake', 'real']))
    print('--'*10, "Logistic Regression Accuracy :", accuracy_score(y_test, lr_pred))
    lr_acc = accuracy_score(y_test, lr_pred)
    print('--'*10, "Logistic Regression F1 Score :", f1_score(y_test, lr_pred))
    lr_f1 = f1_score(y_test, lr_pred)

    report = classification_report(y_test, lr_pred, target_names=['fake', 'real'])
    with open("lr_report.txt", "w") as f:
        f.write(report)

    mlflow.log_param("tfidf_max_features", 5000)
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("accuracy", lr_acc)
    mlflow.log_metric("f1_score", lr_f1)
    mlflow.log_artifact("lr_report.txt")
    mlflow.sklearn.log_model(lr_pipeline, "model")



with mlflow.start_run(run_name="Fake News Detection SVM Model"):
    svc_pipeline.fit(X_train, y_train)
    svc_pred = svc_pipeline.predict(X_test)

    print('--'*10, "SVM Classification Report :")
    print(classification_report(y_test, svc_pred, target_names=['fake', 'real']))
    print('--'*10, "SVM Accuracy :", accuracy_score(y_test, svc_pred))
    svc_acc = accuracy_score(y_test, svc_pred)
    print('--'*10, "SVM F1 Score :", f1_score(y_test, svc_pred))
    svc_f1 = f1_score(y_test, svc_pred)

    report = classification_report(y_test, svc_pred, target_names=['fake', 'real'])
    with open("svc_report.txt", "w") as f:
        f.write(report)


    mlflow.log_param("tfidf_max_features", 5000)
    mlflow.log_param("model_type", "SVM")

    mlflow.log_metric("accuracy", svc_acc)
    mlflow.log_metric("f1_score", svc_f1)
    mlflow.log_artifact("svc_report.txt")

    mlflow.sklearn.log_model(svc_pipeline, "model")
mlflow.end_run()



user_input = input("Enter a news article: ")
predicted_class = svc_pipeline.predict([user_input])[0]
pred_label = 'real' if predicted_class == 1 else 'fake'

print(f"Predicted class: {pred_label}")

import joblib
joblib.dump(svc_pipeline, 'fake_news_svm_model.pkl')


# model = joblib.load('fake_news_svm_model.pkl')
# new_input = input("Enter a news article for prediction: ")