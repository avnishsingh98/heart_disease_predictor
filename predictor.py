import warnings
warnings.filterwarnings("ignore")                       # Ignore warnings
import pandas as pd                                     # For Dataframes
import numpy as np                                      # For Arrays
from sklearn.model_selection import train_test_split    # Splits Data
from sklearn.metrics import accuracy_score              # Grade result
from sklearn.preprocessing import StandardScaler        # Stadardize Data
# Algorithms
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#CONSTANTS
MAX_ITER = 6
TOLERANCE = 1e-3
ETA = 0.001
LOG_REG_C_VAL = 1
SVM_C_VAL = 0.1
MAX_DEPTH = 6
TREES = 10
NEIGHBORS = 3

# Load in the data from the csv file
df = pd.DataFrame.to_numpy(pd.read_csv('heart1.csv'))
# Separate the desired features
X = df[:,:13]
# Extract the classifications
y = df[:,13]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

sc = StandardScaler()                   # create the standard scalar
sc.fit(X_train)                         # compute the required transformation
X_train_std = sc.transform(X_train)     # apply to the training data
X_test_std = sc.transform(X_test)       # and SAME transformation of test data

# vstack puts first array above the second in a vertical stack
# hstack puts first array to left of the second in a horizontal stack
X_combined = np.vstack((X_train, X_test))               # Combined dataset
X_combined_std = np.vstack((X_train_std, X_test_std))   # Combined Standardised dataset
y_combined = np.hstack((y_train, y_test))


###############################################################################
# Perceptron                                                                  #
###############################################################################
def prcptrn():

    ppn = Perceptron(max_iter = MAX_ITER, tol = TOLERANCE, eta0 = ETA, \
                        fit_intercept=True, random_state=0, verbose=False)
    # Do the training
    ppn.fit(X_train_std, y_train)
    # Prediction on combined dataset
    y_combined_pred = ppn.predict(X_combined_std)

    print('\n Perceptron Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))


###############################################################################
# Logistic Regression                                                         #
###############################################################################
def log_reg():

    lr = LogisticRegression(C = LOG_REG_C_VAL, solver='liblinear', \
                            multi_class='ovr', random_state=0)
    # Do the training
    lr.fit(X_train_std, y_train)
    # Prediction on combined dataset
    y_combined_pred = lr.predict(X_combined_std)

    print('\n Logistic Regression Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))


###############################################################################
# Support Vector Machine                                                      #
###############################################################################
def svm():

    svm = SVC(kernel='linear', C = SVM_C_VAL, random_state=0)
    # Do the training
    svm.fit(X_train_std, y_train)
    # Prediction on combined dataset
    y_combined_pred = svm.predict(X_combined_std)

    print('\n Support Vector Machine Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))


###############################################################################
# Decision Tree Learning                                                      #
###############################################################################
def tree():

    tree = DecisionTreeClassifier(criterion='entropy', max_depth = MAX_DEPTH \
                                    ,random_state=0)
    # Do the training
    tree.fit(X_train,y_train)
    # Prediction on combined dataset
    y_combined_pred = tree.predict(X_combined)

    print('\n Decision Tree Learning Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))


###############################################################################
# Random Forest                                                               #
###############################################################################
def forest():

    forest = RandomForestClassifier(criterion='entropy', n_estimators= TREES, \
                                        random_state=1, n_jobs=4)
    # Do the training
    forest.fit(X_train,y_train)
    # Prediction on combined dataset
    y_combined_pred = forest.predict(X_combined)

    print('\n Random Forest Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))


###############################################################################
# K-Nearest Neighbours                                                        #
###############################################################################
def knn():

    knn = KNeighborsClassifier(n_neighbors = NEIGHBORS ,p=2,metric='minkowski')
    # Do the training
    knn.fit(X_train_std,y_train)
    # Prediction on combined dataset
    y_combined_pred = knn.predict(X_combined_std)

    print('\n K-Nearest Neighbours Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))


###############################################################################
# Main Program                                                                #
###############################################################################
prcptrn()           # Perceptron Method
log_reg()           # Logistic Regression Method
svm()               # Support Vector Machine Method
tree()              # Decision Tree Method
forest()            # Random Forest Method
knn()               # K- Nearest Neighbours Method
