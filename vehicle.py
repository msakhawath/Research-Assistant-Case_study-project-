import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest

# Load the dataset
vehicle_claims_df = pd.read_csv('vehicle_claims.csv')

# Data preprocessing
vehicle_claims_df['Runned_Miles'] = pd.to_numeric(vehicle_claims_df['Runned_Miles'].str.replace(',', ''), errors='coerce')
vehicle_claims_df.dropna(subset=['Runned_Miles'], inplace=True)

# Assuming 'repair_cost' is the feature of interest for anomaly detection
vehicle_claims_df['repair_cost'].fillna(vehicle_claims_df['repair_cost'].median(), inplace=True)

# Label anomalies using a simple method
threshold = np.percentile(vehicle_claims_df['repair_cost'], 95)
vehicle_claims_df['anomaly'] = (vehicle_claims_df['repair_cost'] > threshold).astype(int)
print(vehicle_claims_df['anomaly'].value_counts())


# Feature selection and train-test split
X = vehicle_claims_df[['repair_cost']]
y = vehicle_claims_df['anomaly']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the outlier detection models with the fixed contamination rate
fixed_contamination = 0.1
classifiers = {
    'ABOD': ABOD(contamination=fixed_contamination),
    'KNN': KNN(contamination=fixed_contamination),
    'LOF': LOF(contamination=fixed_contamination),
    'One-Class SVM': OCSVM(contamination=fixed_contamination),
    'Isolation Forest': IForest(contamination=fixed_contamination, random_state=5)
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

    results_df = results_df.append({
        'Model': clf_name,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc
    }, ignore_index=True)

# Plotting
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Model', y='Value', hue='Metric', data=pd.melt(results_df, id_vars='Model', var_name='Metric', value_name='Value'))
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 10),
                textcoords = 'offset points')

plt.tight_layout()
plt.show()
