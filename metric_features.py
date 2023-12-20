import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest

# Load datasets
superstore_df = pd.read_excel('superstore.xls')  # Replace with your path
wine_data = pd.read_csv('wine_data.csv')  # Replace with your path

# Process superstore data
y_superstore = (superstore_df['Profit'] < 0).astype(int)
X_superstore = superstore_df.drop('Profit', axis=1).select_dtypes(include=[np.number])
X_superstore = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_superstore)
X_train_superstore, X_test_superstore, y_train_superstore, y_test_superstore = train_test_split(X_superstore, y_superstore, test_size=0.3, random_state=42)

# Process wine data
wine_data['anomaly'] = (wine_data['quality'] < 5).astype(int)
X_wine = wine_data.drop(['quality', 'anomaly'], axis=1)
y_wine = wine_data['anomaly']
X_wine = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_wine)
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

# Define models
fixed_contamination = 0.1
classifiers = {
    'ABOD': ABOD(contamination=fixed_contamination),
    'KNN': KNN(contamination=fixed_contamination),
    'LOF': LOF(contamination=fixed_contamination),
    'One-Class SVM': OCSVM(contamination=fixed_contamination),
    'Isolation Forest': IForest(contamination=fixed_contamination, random_state=5)  # Only for wine
}

# Evaluate models
def evaluate_models(X_train, X_test, y_train, y_test, classifiers):
    results = pd.DataFrame(columns=['Model', 'F1 Score', 'ROC AUC', 'PR AUC'])
    for clf_name, clf in classifiers.items():
        clf.fit(X_train)
        y_pred = clf.predict(X_test)
        y_scores = np.nan_to_num(clf.decision_function(X_test))
        results = results.append({
            'Model': clf_name,
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_scores),
            'PR AUC': average_precision_score(y_test, y_scores)
        }, ignore_index=True)
    return results

results_superstore = evaluate_models(X_train_superstore, X_test_superstore, y_train_superstore, y_test_superstore, classifiers)
results_wine = evaluate_models(X_train_wine, X_test_wine, y_train_wine, y_test_wine, classifiers)

# Calculate number of features
n_features_superstore = X_superstore.shape[1]
n_features_wine = X_wine.shape[1]
# Combine results with feature counts
results_superstore['Dataset'] = 'Superstore'
results_wine['Dataset'] = 'Wine'
results_combined = pd.concat([results_superstore, results_wine])
results_combined['Features'] = results_combined['Dataset'].map({
    'Superstore': n_features_superstore,
    'Wine': n_features_wine
})

# Melt for correlation analysis
melted_results = pd.melt(results_combined, id_vars=['Model', 'Dataset', 'Features'], var_name='Metric', value_name='Value')

# Function to plot correlation
def plot_correlation(data, x_metric, y_metric, title):
    # Filter data for the two metrics
    filtered_data = data[data['Metric'].isin([x_metric, y_metric])]
    # Pivot to get a DataFrame suitable for correlation calculation
    pivot_data = filtered_data.pivot_table(index=['Model', 'Dataset', 'Features'], columns='Metric', values='Value').reset_index()
    # Calculate Spearman correlation
    correlation = pivot_data.corr(method='spearman')[x_metric][y_metric]
    # Plot
    sns.scatterplot(data=pivot_data, x=x_metric, y=y_metric, hue='Dataset')
    plt.title(f'{title}\nSpearman Correlation: {correlation:.2f}')
    plt.show()

# Plot correlations against number of features
plot_correlation(melted_results, 'ROC AUC', 'PR AUC', 'ROC AUC vs PR AUC (Features)')
plot_correlation(melted_results, 'ROC AUC', 'F1 Score', 'ROC AUC vs F1 Score (Features)')
plot_correlation(melted_results, 'PR AUC', 'F1 Score', 'PR AUC vs F1 Score (Features)')
