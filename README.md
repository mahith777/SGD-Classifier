# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1- Split the data into training and testing sets so the model can learn from one portion and be evaluated on another.
2- Train the classifier using Stochastic Gradient Descent, which updates weights step by step with each sample.
3- Predict outcomes on the test set using the trained model.
4- Evaluate performance by calculating accuracy and examining the confusion matrix to see correct vs. incorrect classifications


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Mahith M
RegisterNumber: 212225220061
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix:")
print(cm) 

```

## Output:
<img width="786" height="320" alt="image" src="https://github.com/user-attachments/assets/006cc012-4977-4038-a765-ae92e24b3b35" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
