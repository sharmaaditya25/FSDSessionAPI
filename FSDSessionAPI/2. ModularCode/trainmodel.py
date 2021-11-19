import pandas as pd
from config import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv(datapath)

# Select independent and dependent variable
X = df[rawfields]
y = df[target]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randomstate)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open(modeloutput, "wb"))

#Load the saved model
model = pickle.load(open(modeloutput, "rb"))

#Predictions
output = model.predict(X_test)