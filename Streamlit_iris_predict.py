import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor


def iris_classifier(X_train, y_train, input_data):
    nnclass = MLPClassifier(hidden_layer_sizes=(10, 10,), max_iter=1000 ,random_state=42)
    nnclass.fit(X_train, y_train)
    nnclass_prediction = nnclass.predict(input_data)
    return nnclass_prediction

def iris_regression(X_train, y_train, input_data):
    nnreg = MLPRegressor(hidden_layer_sizes=(10, 10,), max_iter=1000, random_state=42)
    nnreg.fit(X_train, y_train)
    nnreg_prediction = nnreg.predict(input_data)
    return nnreg_prediction

def iris_logregression(X_train, y_train, input_data):
    logreg = LogisticRegression(random_state=42)
    logreg.fit(X_train, y_train)
    logreg_prediction = logreg.predict(input_data)
    return logreg_prediction

def iris_linearregression(X_train, y_train, input_data):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_prediction = lr.predict(input_data)
    return lr_prediction

def iris_svc(X_train, y_train, input_data):
    svm = SVC(random_state=42)
    svm.fit(X_train, y_train)
    svm_prediction = svm.predict(input_data)
    return svm_prediction

def iris_kmeans(X_train, y_train, input_data):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_train)
    kmeans_prediction = kmeans.predict(input_data)
    return kmeans_prediction

def iris_rf(X_train, y_train, input_data):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_prediction = rf.predict(input_data)
    return rf_prediction

def iris_dt(X_train, y_train, input_data):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_prediction = dt.predict(input_data)
    return dt_prediction

def iris_knn(X_train, y_train, input_data):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_prediction = knn.predict(input_data)
    return knn_prediction


def main():
    st.title("Iris Prediction")

    # PREPARING DATA
    iris = load_iris()

    sepal_length = st.sidebar.slider("sepal length (cm)", min_value=4.3, max_value=7.9, step=0.1)
    sepal_width = st.sidebar.slider("sepal width (cm)", min_value=2.0, max_value=4.4, step=0.1)
    petal_length = st.sidebar.slider("petal length (cm)", min_value=1.0, max_value=6.9, step=0.1)
    petal_width = st.sidebar.slider("petal width (cm)", min_value=0.1, max_value=2.5, step=0.1)

    btn_classify = st.sidebar.button("Predict")

    if btn_classify:
        X = iris.data
        Y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        input_data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        input_data = scaler.transform(input_data)

        # Predictions

        P1 = iris_classifier(X_train, y_train, input_data)
        P2 = iris_regression(X_train, y_train, input_data)
        P3 = iris_logregression(X_train, y_train, input_data)
        P4 = iris_linearregression(X_train, y_train, input_data)
        P5 = iris_svc(X_train, y_train, input_data)
        P6 = iris_kmeans(X_train, y_train, input_data)
        P7 = iris_rf(X_train, y_train, input_data)
        P8 = iris_dt(X_train, y_train, input_data)
        P9 = iris_knn(X_train, y_train, input_data)

        st.subheader("Classifier")
        st.text(iris.target_names[int(P1)])
        if P1 == 0:
            st.image("setosa.jpeg")
        elif P1 == 1:
            st.image("versicolor.jpeg")
        elif P1 == 2:
            st.image("virginica.jpeg")

        st.subheader("Regression")
        P2 = int(np.round(P2))
        if P2 > 2: P2 = 2
        st.text(iris.target_names[P2])
        if P2 == 0:
            st.image("setosa.jpeg")
        elif P2 == 1:
            st.image("versicolor.jpeg")
        elif P2 == 2:
            st.image("virginica.jpeg")

        st.subheader("Logistic Regression")
        st.text(iris.target_names[int(P3)])
        if P3 == 0:
            st.image("setosa.jpeg")
        elif P3 == 1:
            st.image("versicolor.jpeg")
        elif P3 == 2:
            st.image("virginica.jpeg")

        st.subheader("Linear Regression")
        P4 = int(np.round(P4))
        if P4 > 2: P4 = 2
        st.text(iris.target_names[P4])
        if P4 == 0:
            st.image("setosa.jpeg")
        elif P4 == 1:
            st.image("versicolor.jpeg")
        elif P4 == 2:
            st.image("virginica.jpeg")

        st.subheader("SVC")
        st.text(iris.target_names[int(P5)])
        if P5 == 0:
            st.image("setosa.jpeg")
        elif P5 == 1:
            st.image("versicolor.jpeg")
        elif P5 == 2:
            st.image("virginica.jpeg")

        st.subheader("Kmeans")
        st.text(iris.target_names[int(P6)])
        if P6 == 0:
            st.image("setosa.jpeg")
        elif P6 == 1:
            st.image("versicolor.jpeg")
        elif P6 == 2:
            st.image("virginica.jpeg")

        st.subheader("Randon Forest")
        st.text(iris.target_names[int(P7)])
        if P7 == 0:
            st.image("setosa.jpeg")
        elif P7 == 1:
            st.image("versicolor.jpeg")
        elif P7 == 2:
            st.image("virginica.jpeg")

        st.subheader("Decision Tree")
        st.text(iris.target_names[int(P8)])
        if P8 == 0:
            st.image("setosa.jpeg")
        elif P8 == 1:
            st.image("versicolor.jpeg")
        elif P8 == 2:
            st.image("virginica.jpeg")

        st.subheader("KNN")
        st.text(iris.target_names[int(P9)])
        if P9 == 0:
            st.image("setosa.jpeg")
        elif P9 == 1:
            st.image("versicolor.jpeg")
        elif P9 == 2:
            st.image("virginica.jpeg")


if __name__ == "__main__":
    main()