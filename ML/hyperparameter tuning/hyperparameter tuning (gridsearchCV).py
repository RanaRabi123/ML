import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt

df = datasets.load_digits()


            # remember to pass the parameters values in list , otherwise it will throw error 
model_params = {
    'DTC' : {
        'model' : DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini']
        }
    },
    'RFC' : {
        'model' : RandomForestClassifier(),
        'params' : {
            'criterion' : ['gini'],
            'ccp_alpha' : [0.1]
        }
    }, 
    'LR' : {
        'model' : LogisticRegression(solver='saga'),
        'params' : {
            'max_iter' : [400],
            'penalty' : ['elasticnet'],
            'l1_ratio' : [0.5] 
        }
    },
    'SVM' : {
        'model' : SVC(),
        'params' : {
            'kernel' : ['linear'],
            'gamma' : ['auto']
        }
    }, 
    'GNB' : {
        'model' : GaussianNB(),
        'params' : {}
    }
}

score = []
for model_name, model_parameters in model_params.items():
    clf = GridSearchCV(model_parameters['model'], model_parameters['params'], cv=4, return_train_score=False, verbose=1)
    clf.fit(df.data, df.target)

        #  run it to see the properties of the clf , if you want ot print something else 
    print(dir(clf))

    score.append({
        'model': model_name,
        'best_model_parameters' : clf.best_params_,
        "Best_model_score" : clf.best_score_
    })

score_df = pd.DataFrame(score, columns=['model', 'best_model_parameters', 'Best_model_score'])
print(score_df)



            #  plottting
clrs = ['red' if x==max(score_df.Best_model_score) else 'grey' for x in score_df.Best_model_score]
sns.barplot(x=score_df.model, y=score_df.Best_model_score, width=0.5, palette=clrs)
plt.savefig('model perfromance.jpg', dpi=300, bbox_inches='tight')
plt.show()