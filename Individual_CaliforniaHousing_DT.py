import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

#------------------------------------------------------------
# STEP 1 : Load the Data
#------------------------------------------------------------

df = pd.read_csv("california_housing.csv")
print("Shape of Dataset : ",df.shape)

print("First few records : ")
print() 
print(df.head(10))

#------------------------------------------------------------
# STEP 2 : Seperate Features and Labels
#------------------------------------------------------------

X = df.drop("target", axis = 1)
Y = df["target"]

print("Shape of X : ", X.shape)
print("Shape of Y : ", Y.shape)

#------------------------------------------------------------
# STEP 3 : Split Dataset for Training and Testing
#------------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.2, random_state = 42)

#------------------------------------------------------------
# STEP 4 : Create the Model 
#------------------------------------------------------------

model = DecisionTreeRegressor(random_state = 42)

#------------------------------------------------------------
# STEP 5 : Train the Model
#------------------------------------------------------------

model = model.fit(X_train, Y_train)

#------------------------------------------------------------
# STEP 6 : Test the Model
#------------------------------------------------------------

Y_pred = model.predict(X_test)

#------------------------------------------------------------
# STEP 7 : Evaluate the Model
#------------------------------------------------------------

print("MSE : ", mean_squared_error(Y_test, Y_pred))
print("R2 : ", r2_score(Y_test, Y_pred))
