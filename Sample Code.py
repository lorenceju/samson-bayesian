#importing libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Productivity Dataset 1.csv")

# Define predictors (X) and outcome (y)
X = data[["Sleep Hours", "Caffeine Intake", "Stress Levels"]]
y = data["Productivity Score"]

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Create the regression model
model = LinearRegression()

#Train the model
model.fit(X_train,y_train)

#Predict productivity for the test data
y_pred = model.predict(X_test)

#Evaluate the model
r2 = r2_score(y_test, y_pred)

print("R-squared score:",r2)

#Show model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Sleep Hours:", model.coef_[0])
print("Caffeine_mg:", model.coef_[1])
print("Stress_Level:", model.coef_[2])

