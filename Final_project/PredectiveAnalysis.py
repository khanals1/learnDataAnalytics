import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def encodeCols(df):
    categorical_columns = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
                           "admission_source_id", "time_in_hospital", "num_lab_procedures",
                           "num_procedures", "num_medications", "number_outpatient",
                           "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
                           "diabetesMed"]
    df = pd.get_dummies(df, columns=categorical_columns)
    df = pd.get_dummies(df, columns=['readmitted'], drop_first=True)
    X = df.drop(['readmitted_Yes'], axis=1)
    y = df['readmitted_Yes']
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

    logreg = LogisticRegression(C=1)
    logreg.fit(X_train, y_train)
    y_prediction = logreg.predict(X_test)
    results = metrics.confusion_matrix(y_test, y_prediction)
    model_accuracy = metrics.classification_report(y_test, y_prediction)
    print("For Logistic Regression: \n")
    print(results)
    print(model_accuracy)

    return df
