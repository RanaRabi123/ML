import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Housing.csv')

area_list = [random.randrange(5000, 15000) for _ in range(10)]

model = LinearRegression()
model.fit(df[['area']], df.price)

sns.scatterplot(x=df.area, y=df.price)  # grpah of the data

prediction = model.predict([[i] for i in area_list])
# print(prediction)

sns.lineplot(x=df.area, y=model.predict(df[['area']]), color='r')   # predcition line grpah on training_data
sns.scatterplot(x=area_list, y=prediction, color='y' ,markers='*')  # prediction scatter_points grpah 

plt.savefig("housing_plot.png", dpi=300, bbox_inches="tight")   # it will save the plot
plt.show()