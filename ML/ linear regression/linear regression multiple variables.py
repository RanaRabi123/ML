import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from word2number import w2n
import random

df = pd.read_csv('E:\intern preparation\ML\ linear regression\hiring.csv')

    # coverting numeric words to numbers 
df.experience = df.experience.apply(lambda x: w2n.word_to_num(x) if isinstance(x, str) else x)

    # handling NAn values 
median_experience = int(df.experience.median())
median_test = df['test_score(out of 10)'].median()
median_interview = df['interview_score(out of 10)'].median()
median_salary = df['salary($)'].median()
    # replacing the NAN values with their median value 
df.experience = df.experience.fillna(median_experience)
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test)
df['interview_score(out of 10)'] = df['interview_score(out of 10)'].fillna(median_interview)
df['salary($)'] = df['salary($)'].fillna(median_salary)

    # making random data
experience_list = [random.randrange(1, 15) for _ in range(8)]
test_list = [random.randrange(1, 10) for _ in range(8)]
interview_list = [random.randrange(1, 10) for _ in range(8)]

x_train = df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']]
y_train = df['salary($)']
x_test = list(zip(experience_list, test_list, interview_list))

model = LinearRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)

print("coefficient of the model is : ", model.coef_)
print("intercept of the model is : ", model.intercept_)


    # actual VS predicted 
sns.scatterplot(x=y_train, y=model.predict(x_train))
sns.lineplot(x=y_train, y=model.predict(x_train))
plt.savefig("hiring_plot.png", dpi=300, bbox_inches="tight")
plt.show()