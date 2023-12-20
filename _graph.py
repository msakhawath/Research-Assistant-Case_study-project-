import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
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

# Remove rows with NaN values
wine_data.dropna(inplace=True)

# Define the target variable
wine_data['anomaly'] = (wine_data['quality'] < 5).astype(int)

# Use all other features for anomaly detection
X = wine_data.drop(['quality', 'anomaly'], axis=1)
y = wine_data['anomaly']

# Normalize the features
minmax = MinMaxScaler(feature_range=(0, 1))
X_normalized = minmax.fit_transform(X)
X = pd.DataFrame(X_normalized, columns=X.columns)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fixed contamination rate
fixed_contamination = 0.1

# Define the outlier detection models
classifiers = {
    'ABOD': ABOD(contamination=fixed_contamination),
    'KNN': KNN(contamination=fixed_contamination),
    'LOF': LOF(contamination=fixed_contamination),
    'OCSVM': OCSVM(contamination=fixed_contamination),
    'IForest': IForest(contamination=fixed_contamination, random_state=0),
    'FeatureBagging': FeatureBagging(contamination=fixed_contamination, random_state=0)
}

# Prepare DataFrame to store predictions and feature values
outlier_predictions = pd.DataFrame()

# Iterate over classifiers to get predictions
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    # Get the outlier scores (the lower, the more normal)
    scores = clf.decision_function(X_test)
    # Normalize the scores using RobustScaler to mitigate the effect of outliers
    scores_scaled = RobustScaler().fit_transform(scores.reshape(-1, 1)).flatten()
    # Predict inliers (0) and outliers (1)
    predictions = clf.predict(X_test)
    # Store results
    df_results = pd.DataFrame({
        'Model': [clf_name]*len(scores),
        'Outlier Score': scores_scaled,
        'Prediction': predictions
    })
    # Append to the outlier_predictions DataFrame
    outlier_predictions = pd.concat([outlier_predictions, df_results], ignore_index=True)

# Now create the boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=outlier_predictions, x='Model', y='Outlier Score', hue='Prediction')
plt.title('Combined Boxplot for Inliers and Outliers Across Models')
plt.xlabel('Model')
plt.ylabel('Normalized Outlier Score')
plt.legend(title='Prediction', labels=['Inliers', 'Outliers'])
plt.tight_layout()
plt.show()
