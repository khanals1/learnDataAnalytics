import pandas as pd
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def prediction():
    adults_dF = read_csv("adultsData.csv", na_values='?')
    adults_dF = adults_dF.dropna()
    conditions = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex',
                  'hoursperweek']
    adults_dF = pd.get_dummies(adults_dF, columns=conditions)
    adults_dF = pd.get_dummies(adults_dF, columns=['class'], drop_first=True)
    X = adults_dF.drop(['class_ >50K'], axis=1)
    y = adults_dF['class_ >50K']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train)
    y_prediction = decision_tree_classifier.predict(X_test)
    results = metrics.confusion_matrix(y_test, y_prediction)
    model_accuracy = metrics.classification_report(y_test, y_prediction)
    print("For Decision Tree Classifier: \n")
    print(results)
    print(model_accuracy)

    gaussian_model = GaussianNB()
    gaussian_model.fit(X_train, y_train)
    y_prediction = gaussian_model.predict(X_test)
    results = metrics.confusion_matrix(y_test, y_prediction)
    model_accuracy = metrics.classification_report(y_test, y_prediction)
    print("For Naive Bayes Model: \n")
    print(results)
    print(model_accuracy)


if __name__ == '__main__':
    prediction()

"""
For Decision Tree Classifier: 

[[4360  609]
 [ 514 1029]]
              precision    recall  f1-score   support

           0       0.89      0.88      0.89      4969
           1       0.63      0.67      0.65      1543

    accuracy                           0.83      6512
   macro avg       0.76      0.77      0.77      6512
weighted avg       0.83      0.83      0.83      6512

For Naive Bayes Model: 

[[4719  250]
 [1016  527]]
              precision    recall  f1-score   support

           0       0.82      0.95      0.88      4969
           1       0.68      0.34      0.45      1543

    accuracy                           0.81      6512
   macro avg       0.75      0.65      0.67      6512
weighted avg       0.79      0.81      0.78      6512

"""