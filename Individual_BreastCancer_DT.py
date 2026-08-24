# dt = decision tree
# this is unimodel
# this code is in scripting format, no main no starter

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#-------------------------------------
# STEP 1 : Load the Dataset 
#-------------------------------------

df = pd.read_csv("breast_cancer.csv")

print("Shape of Dataset : ", df.shape)
print("First 5 Records : ")
print(df.head())

#----------------------------------------
# STEP 2 : Seperate Features and Labels 
#----------------------------------------

X = df.drop("target", axis = 1)
Y = df["target"]

print("Shape of X : ", X.shape)
print("Shape of Y : ", Y.shape)

#--------------------------------------------------
# STEP 3 : Split Dataset for Training and Testing
#--------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


#----------------------------------------
# STEP 4 : Scale the Features 
#----------------------------------------

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)
# We don't scale Ytrain and Ytest because it contains 0 and 1 or such data generally which doesn't need to be scaled

#----------------------------------------
# STEP 5 : Create the Model 
#----------------------------------------

model = DecisionTreeClassifier(random_state = 42)

#----------------------------------------
# STEP 6 : Train the Model 
#----------------------------------------

model = model.fit(X_train, Y_train)

#----------------------------------------
# STEP 7 : Test the Model 
#----------------------------------------

Y_pred = model.predict(X_test)


#----------------------------------------
# STEP 8 : Evaluate the Model 
#----------------------------------------

print("Accuarcy Score is : ")
print(accuracy_score(Y_test, Y_pred))

print("Confusion Matrix : ")
print(confusion_matrix(Y_test, Y_pred))