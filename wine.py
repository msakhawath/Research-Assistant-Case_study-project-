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

# Load the dataset (Assuming it is loaded as 'wine_data')
wine_data = pd.read_csv('wine_data.csv')

wine_data.info()

# Define the target variable (anomalies as quality below 5)
wine_data['anomaly'] = (wine_data['quality'] < 5).astype(int)

# Use all other features for anomaly detection
X = wine_data.drop(['quality', 'anomaly'], axis=1)
y = wine_data['anomaly']

# Normalize the features
minmax = MinMaxScaler(feature_range=(0, 1))
X = minmax.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fixed contamination rate
fixed_contamination = 0.1

# Define the outlier detection models with the fixed contamination rate
classifiers = {
    'ABOD': ABOD(contamination=fixed_contamination),
    'KNN': KNN(contamination=fixed_contamination),
    'LOF': LOF(contamination=fixed_contamination),
    'One-Class SVM': OCSVM(contamination=fixed_contamination),
    'Isolation Forest': IForest(contamination=fixed_contamination, random_state=5),
    'Feature Bagging': FeatureBagging(contamination=fixed_contamination, random_state=5)
}

# DataFrame to store results
results_df = pd.DataFrame(columns=['Model', 'F1 Score', 'ROC AUC', 'PR AUC'])

# Iterate over classifiers and calculate metrics
for clf_name, clf in classifiers.items():
    clf.fit(X_train)
    y_pred = clf.predict(X_test)
    y_scores = np.nan_to_num(clf.decision_function(X_test))

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)

    # Append results to DataFrame
    results_df = results_df.append({
        'Model': clf_name,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc
    }, ignore_index=True)

# Plotting
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Model', y='Value', hue='Metric', data=pd.melt(results_df, id_vars='Model', var_name='Metric', value_name='Value'))
plt.title('Model Performance Comparison for wine data')
plt.xticks(rotation=45)

# Adding annotations
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 10),
                textcoords = 'offset points')

plt.tight_layout()
plt.show()
