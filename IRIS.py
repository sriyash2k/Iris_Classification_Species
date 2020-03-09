# Data Preprocessing

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Iris.csv')

x=dataset.iloc[:, [1,2,3,4]].values
y=dataset.iloc[:, 5].values

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
y=encoder.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Data Visualization

plt.scatter(x[1:50, 1],x[1:50, 2],color='red')
plt.scatter(x[50:100 , 1],x[50:100 ,2],color='green')
plt.scatter(x[100:150 , 1],x[100:150 ,2],color='yellow')


# Model Creation

from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=42)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)


# Results

from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)*100
print(score)