import regex as re
import gzip
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def preprocessing():
    with gzip.open("/content/finefoods.txt.gz", "rt", encoding='latin-1') as f:
        file_content = f.read()

    score = []
    text = []
    for str in file_content.strip().split('\n'):
        if 'review/score:' in str:
            score.append(int(float(re.findall(r'(?<=:\s).*', str)[0])))
        elif 'review/text' in str:
            text.append(re.findall(r'(?<=:\s).*', str)[0])

    x_train, x_test, y_train, y_test  = train_test_split(text, score, random_state=33, test_size=0.22)

    return x_train, x_test, y_train, y_test
     

x_train, x_test, y_train, y_test = preprocessing()

lr_pipeline = Pipeline([
    ('vectorizer' , TfidfVectorizer()) ,
    ('lr_model', LogisticRegression(max_iter=1000)),
    ])

svc_pipeline = Pipeline([
    ('vectorizer' , TfidfVectorizer()) ,
    ('svc_model', LinearSVC())  # for textual one linear is better , because it does not match similairyt score for each point with n_points 
    ])

m_nb_pipeline = Pipeline([
    ('vectorizer' , TfidfVectorizer()) ,
    ('m_nb_model', MultinomialNB())
    ])



print('\n--'*10, 'LogisticRegression Model', '--'*10, '\n')
lr_pipeline.fit(x_train, y_train)
lr_pred =lr_pipeline.predict(x_test)
lr_cm = confusion_matrix(y_test, lr_pred)

print('\n--'*10, 'Support Vector Classifier Model', '--'*10, '\n')
svc_pipeline.fit(x_train, y_train)
svc_pred =svc_pipeline.predict(x_test)
svc_cm = confusion_matrix(y_test, svc_pred)

print('\n--'*10, 'MultinomialNB Model', '--'*10, '\n')
m_nb_pipeline.fit(x_train, y_train)
m_nb_pred =m_nb_pipeline.predict(x_test)
m_nb_cm = confusion_matrix(y_test, m_nb_pred)


cm_dict = {
    'lr_cm':lr_cm,
    'svc_cm':svc_cm,
    'm_nb_cm':m_nb_cm
}
for key, value in cm_dict.items():
    sns.heatmap(value, annot=True, fmt='d')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(key)
    plt.show()
