#Add Key imports here!
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math


TRAIN_URL = "https://raw.githubusercontent.com/karkir0003/DSGT-Bootcamp-Material/main/Udemy%20Material/Airline%20Satisfaction/train.csv"
def read_train_dataset():
    """
    This function should read in the train.csv and return it 
    in whatever representation you like
    """
    df = pd.read_csv(TRAIN_URL)
    return df


    ###YOUR CODE HERE####
    raise NotImplementedError("Did not implement read_train_dataset() function")
    #####################

def preprocess_dataset(dataset):
    """
    Given the raw dataset read in from your read_train_dataset() function,
    
    process the dataset accordingly 
    """

    df = read_train_dataset()

    gender = []
    cust = []
    travel = []
    classType = []
    satisfaction = []

    for ind in df.index:
        if (df['Gender'][ind] == 'Female'):
            gender.append(0)
        else:
            gender.append(1)
        
        if (df['Customer Type'][ind] == 'Loyal Customer'):
            cust.append(0)
        else:
            cust.append(1)
        
        if (df['Type of Travel'][ind] == 'Personal Travel'):
            travel.append(0)
        else:
            travel.append(1)
        
        if (df['Class'][ind] == 'EcoPlus'):
            classType.append(0)
        else:
            classType.append(1)

        if (df['satisfaction'][ind] == 'neutral or dissatisfied'):
            satisfaction.append(0)
        else:
            satisfaction.append(1)

    df['Gender'] = gender
    df['Customer Type'] = cust
    df['Type of Travel'] = travel
    df['Class'] = classType
    df['satisfaction'] = satisfaction

    return df

def train_model():
    """
    Given your cleaned data, train your Machine Learning model on it and return the
    model
    
    MANDATORY FUNCTION TO IMPLEMENT
    """

    df = preprocess_dataset(read_train_dataset)

    X = df.iloc[1: , :]
    X = X.iloc[:, :-2]

    y = df.iloc[1: , :]
    y = y.iloc[:, -1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

    dt = DecisionTreeClassifier(max_leaf_nodes = 5)
    svc = SVC(kernel = 'linear')
    mlp = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, max_iter = 100)
    knn = KNeighborsClassifier(n_neighbors=15)

    dt.fit(X_train, y_train)
    svc.fit(X_train, y_train.values.ravel())
    mlp.fit(X_train, y_train.values.ravel())
    knn.fit(X_train, y_train.values.ravel())
    
    predD = dt.predict(X_test)
    predS = svc.predict(X_test)
    predM = mlp.predict(X_test)
    predK = knn.predict(X_test)

    accD = accuracy_score(y_test, predD)
    accS = accuracy_score(y_test, predS)
    accM = accuracy_score(y_test, predM)
    accK = accuracy_score(y_test, predK)

    print(accD)
    print(accS)
    print(accM)
    print(accK)