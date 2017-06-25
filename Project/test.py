from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------------
def read_file(file_name):
    return pd.read_csv(file_name)


def label_encoder(file):
    file = file.drop('e.1', 1)

    le = LabelEncoder()

    for col in file.columns:
        file[col] = le.fit_transform(file[col])

    return file


def train_and_test_set(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)


def separate_set(file):
    return file.iloc[:, 1:23], file.iloc[:, 0]
# ---------------------------------------------------------------------------------
# --------------------------  SUPPORT VECTOR MACHINE ------------------------------


def support_vector_machine(x_train, x_test, y_train, y_test):
    svm_model = SVC(kernel='rbf', C=1, gamma=0.1).fit(x_train, y_train)
    y_predicted = svm_model.predict(x_test)

    print("SVM:")
    print("\tf1_score: " + str(metrics.f1_score(y_test, y_predicted)))
    print("-------------------------------------")
# ---------------------------------------------------------------------------------
# ------------------------------  NAIVE BAYES -------------------------------------


def naive_bayes(x_train, x_test, y_train, y_test):
    nb_model = GaussianNB().fit(x_train, y_train)
    # positive class prediction probabilities
    y_prob = nb_model.predict_proba(x_test)[:, 1]
    y_predicted = np.where(y_prob > 0.5, 1, 0)

    print("NAIVE BAYES")
    print("\tf1_score: " + str(metrics.f1_score(y_test, y_predicted)))
    print("-------------------------------------")
# ---------------------------------------------------------------------------------
# -------------------------- K NEIGHBORS CLASSIFIER  ------------------------------


def k_neighbors_classifier(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski').fit(x_train, y_train)
    y_predicted = knn.predict(x_test)

    print("K NEIGHBORS CLASSIFIER:")
    print("\tf1_score: " + str(metrics.f1_score(y_test, y_predicted)))
    print("-------------------------------------")
# ---------------------------------------------------------------------------------


if __name__ == '__main__':
    data = label_encoder(read_file('data/train.csv'))
    x, y = separate_set(data)
    x_train, x_test, y_train, y_test = train_and_test_set(x, y)

    # --------- SVM ---------
    support_vector_machine(x_train, x_test, y_train, y_test)

    # --------- NB ----------
    naive_bayes(x_train, x_test, y_train, y_test)

    # --------- KNN ---------
    k_neighbors_classifier(x_train, x_test, y_train, y_test)