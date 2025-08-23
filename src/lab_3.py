"""
This is a lab for Performing Python Lab 3: Logistic Regression for Coursera AI Course
"""

"""
Imports
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss

import warnings
warnings.filterwarnings('ignore')

"""
Main Block
"""

def main():
    #churn_df = pd.read_csv("ChurnData.csv")
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"

    churn_df = pd.read_csv(url)
    # Verify Sample
    churn_df.sample(5)
    # Explore and Select Features
    churn_df.describe()

    # Data Preprocessing
    churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip','churn',]
    churn_df['churn'] = churn_df['churn'].astype('int')
                        
    # Model X and y
    X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',]])
    X[0:5] # print the first 5 values

    y = np.asarray(churn_df['churn'])
    y[0:5] # Print the first 5 values
    
    # Normalize Data
    X_norm = StandarScaler().fit(x).transform(X)
    X_norm[0:5] # Print the first 5 values

    # Split the Dataset
    X_train, X_test, y_train, y_test = train_split( X_norm, test_size=0.2, random_state=4)

    # Logistic Regression Classifier Modeling
    LR = LogisticRegression().fit(X_train, y_train)

    yhat = LR.predict(X_test)
    yhat[:10]

    yhat_prob = LR.predict_proba(X_test)
    yhat_prob[:10]

    # Gather coefficients and Plot
    # Large Positive LR Coefficients indicate Parameter Increase
    # Large Negative LR Coefficients indicate opposite
    # Small absolute value indicates weaker affect from the change
    coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
    coefficients.sort_values().plot(kind='barh')
    plt.title("Feature Coefficients in Logistic Regression Churn Model")
    plt.xlabel("Coefficient Value")
    plt.show()

    # Evaluate Performance - Log-Loss
    log_loss(y_test, yhat_prob)

if __name__=="__main__":
    main()
