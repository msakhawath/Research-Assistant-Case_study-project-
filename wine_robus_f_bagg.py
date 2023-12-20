import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, make_scorer
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.feature_bagging import FeatureBagging
import matplotlib.pyplot as plt
import seaborn as sns

# Custom cross-validation function
from sklearn.utils.validation import check_array

def evaluate_model_cv(model, X, y, metric_func, metric_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Pre-check to ensure no invalid values are in the training set
        X_train = check_array(X_train, force_all_finite=True, ensure_2d=True)
        y_train = check_array(y_train, force_all_finite=True, ensure_2d=False)

        model.fit(X_train)

        # Use the appropriate method to get the scores
        if hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_test)
        elif hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            y_scores = model.predict(X_test)

        # Check if metric_func requires binary predictions and not scores
        if metric_name == 'F1 Score':
            y_scores = np.where(y_scores > 0, 1, 0)

        # Handle NaNs and infinities in y_scores
        y_scores = np.nan_to_num(y_scores, nan=np.nanmean(y_scores), posinf=np.nanmax(y_scores), neginf=np.nanmin(y_scores))

        # Calculate metric
        score = metric_func(y_test, y_scores)
        scores.append(score)

    return np.mean(scores), np.std(scores)


# Load the dataset
wine_data = pd.read_csv('wine_data.csv')

# Define the target variable (anomalies as quality below 5)
wine_data['anomaly'] = (wine_data['quality'] < 5).astype(int)

# Use all other features for anomaly detection
X = wine_data.drop(['quality', 'anomaly'], axis=1).values
y = wine_data['anomaly'].values

# Normalize the features using RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Fixed contamination rate
fixed_contamination = 0.1

# Define the outlier detection models with the fixed contamination rate
classifiers = {
    'ABOD': ABOD(contamination=fixed_contamination),
    'KNN': KNN(contamination=fixed_contamination),
    'LOF': LOF(contamination=fixed_contamination),
    'OCSVM': OCSVM(contamination=fixed_contamination),
    'Isolation Forest': IForest(contamination=fixed_contamination, random_state=42),
    'Feature Bagging': FeatureBagging(contamination=fixed_contamination, random_state=42)
}

# DataFrame to store results
results = []

# Metrics to be evaluated
metrics = {
    'F1 Score': f1_score,
    'ROC AUC': roc_auc_score,
    'PR AUC': average_precision_score
}

# Iterate over classifiers and calculate metrics using custom cross-validation
for clf_name, clf in classifiers.items():
    print(f"Evaluating {clf_name}...")
    clf_metrics = {'Model': clf_name}
    for metric_name, metric_func in metrics.items():
        mean_score, std_score = evaluate_model_cv(clf, X_scaled, y, metric_func, metric_name)
        clf_metrics[f'{metric_name} (mean)'] = mean_score
        clf_metrics[f'{metric_name} (std)'] = std_score
    results.append(clf_metrics)

results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 8))
mean_scores_df = results_df.drop(columns=[col for col in results_df.columns if 'std' in col])
melted_df = pd.melt(mean_scores_df, id_vars='Model', var_name='Metric', value_name='Value')
ax = sns.barplot(x='Model', y='Value', hue='Metric', data=melted_df)
plt.title('Model Performance Comparison for Wine Data')
plt.xticks(rotation=45)

# Adding annotations for the mean scores
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2, height),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()
