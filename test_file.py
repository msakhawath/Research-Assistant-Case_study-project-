import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.feature_bagging import FeatureBagging
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
wine_data = pd.read_csv('wine_data.csv')

# Define the target variable
wine_data['anomaly'] = (wine_data['quality'] < 5).astype(int)

# Features for anomaly detection
X = wine_data.drop(['quality', 'anomaly'], axis=1)
y = wine_data['anomaly']

# Normalize features
minmax = MinMaxScaler(feature_range=(0, 1))
X = minmax.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fixed contamination rate
fixed_contamination = 0.1

# Outlier detection models
classifiers = {
    'ABOD': ABOD(contamination=fixed_contamination),
    'KNN': KNN(contamination=fixed_contamination),
    'LOF': LOF(contamination=fixed_contamination),
    'One-Class SVM': OCSVM(contamination=fixed_contamination),
    'Isolation Forest': IForest(contamination=fixed_contamination, random_state=0),
    'Feature Bagging': FeatureBagging(contamination=fixed_contamination, random_state=0)
}

# Print number of inliers and outliers for each model
for clf_name, clf in classifiers.items():
    clf.fit(X_train)
    y_pred = clf.predict(X_test)

    n_inliers = sum(y_pred == 0)
    n_outliers = sum(y_pred == 1)

    print('Model: ', clf_name)
    print('Number of inliers: ', n_inliers)
    print('Number of outliers: ', n_outliers)
    print('-' * 30)


import pandas as pd

# Define the data
data = {
    'Model': ['ABOD', 'KNN', 'LOF', 'One-Class SVM', 'Isolation Forest', 'Feature Bagging'],
    'Number of inliers': [1300, 1132, 1150, 1146, 1135, 1131],
    'Number of outliers': [0, 168, 150, 154, 165, 169]
}

# Create DataFrame
results_df = pd.DataFrame(data)

# Display the table
print(results_df)
# Save to a CSV file
results_df.to_csv('model_results.csv', index=False)

