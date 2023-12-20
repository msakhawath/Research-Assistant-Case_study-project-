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

# Use all other features for anomaly detection
X = wine_data.drop(['quality', 'anomaly'], axis=1)
y = wine_data['anomaly']

# Normalize the features
minmax = MinMaxScaler(feature_range=(0, 1))
X = minmax.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fixed contamination rate
fixed_contamination = 0.1

# Define the outlier detection models
classifiers = {
    'ABOD': ABOD(contamination=fixed_contamination),
    'KNN': KNN(contamination=fixed_contamination),
    'LOF': LOF(contamination=fixed_contamination),
    'One-Class SVM': OCSVM(contamination=fixed_contamination),
    'Isolation Forest': IForest(contamination=fixed_contamination, random_state=0),
    'Feature Bagging': FeatureBagging(contamination=fixed_contamination, random_state=0)
}

# Set up plot for combined inliers/outliers
fig, axes = plt.subplots(nrows=len(classifiers), ncols=1,
                         figsize=(8, 2 * len(classifiers)))  # Adjusted for smaller size

# Colors for inliers and outliers
inlier_color = 'blue'
outlier_color = 'orange'

# Iterate over classifiers
for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X_train)
    y_pred = clf.predict(X_test)

    # Count inliers and outliers
    inliers_count = np.sum(y_pred == 0)
    outliers_count = np.sum(y_pred == 1)

    # Plot inliers and outliers with smaller point size
    axes[i].scatter(X_test[y_pred == 0][:, 0], X_test[y_pred == 0][:, 1], color=inlier_color, s=20, alpha=0.7,
                    label='Inliers')
    axes[i].scatter(X_test[y_pred == 1][:, 0], X_test[y_pred == 1][:, 1], color=outlier_color, s=20, alpha=0.7,
                    label='Outliers')

    axes[i].set_title(f'{clf_name}', fontsize=10)
    axes[i].legend(prop={'size': 8})

    # Add grid lines
    axes[i].grid(True)

    # Annotate number of inliers and outliers with smaller font
    axes[i].text(0.95, 0.05, f'Inliers: {inliers_count}\nOutliers: {outliers_count}',
                 fontsize=8, transform=axes[i].transAxes, horizontalalignment='right',
                 verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5))

# Adjust layout to fit more tightly
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
