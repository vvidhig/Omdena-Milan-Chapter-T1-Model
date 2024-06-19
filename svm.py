import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn import metrics
import optuna
from xgboost import XGBClassifier
from sklearn.svm import SVC
import mlflow
import mlflow.sklearn
import os
from matplotlib import pyplot as plt

# Set tracking URI and experiment
# mlflow.set_tracking_uri("http://192.168.0.1:5000")
mlflow.set_experiment("Urban_Farming_Prediction_Zone4")

# Load the dataset
dataset = pd.read_csv("dataset/Merged_2014.csv")
numerical_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('Suitable_Areas')

categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()

# Filling categorical columns with mode
for col in categorical_cols:
    mode_value = dataset[col].mode()[0]
    dataset[col].fillna(mode_value, inplace=True)

# Filling Numerical columns with median
for col in numerical_cols:
    median_value = dataset[col].median()
    dataset[col].fillna(median_value, inplace=True)

# Take care of outliers
dataset[numerical_cols] = dataset[numerical_cols].apply(lambda x: x.clip(lower=x.quantile(0.05), upper=x.quantile(0.95)))

log_columns = ["NDVI", "LST", "NDBI", "NDWI", "Roughness", "SAVI", "Slope", "SMI", "solar_radiation"]

# Apply log transformation
for col in log_columns:
    dataset[col] = dataset[col].apply(lambda x: np.log(x) if x > 0 else x)

# Label encoding categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])

# Train test split
X = dataset.drop(columns=['Suitable_Areas'])
y = dataset['Suitable_Areas']
RANDOM_SEED = 6

print("Y value counts:", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)

print("Y_Train value counts:", y_train.value_counts())

# Define XGBClassifier with provided parameters
xgb_params = {
    'n_estimators': 1050,
    'max_depth': 50,
    'learning_rate': 0.01,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'verbosity': 0,
    'booster': 'gbtree'
}

xgb_model = XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train)

# Train SVM model
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Model evaluation metrics
def eval_metrics(actual, pred, pred_proba=None):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    precision = metrics.precision_score(actual, pred, pos_label=1)
    recall = metrics.recall_score(actual, pred, pos_label=1)
    if pred_proba is not None:
        fpr, tpr, _ = metrics.roc_curve(actual, pred_proba)
        auc = metrics.auc(fpr, tpr)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f' % auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('False Positive Rate', size=14)
        plt.ylabel('True Positive Rate', size=14)
        plt.legend(loc='lower right')
        # Save plot
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/ROC_curve.png")
        # Close plot
        plt.close()
    else:
        auc = float('nan')
    return accuracy, f1, precision, recall, auc

def mlflow_logging(model, X, y, name, use_proba=False):
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        if use_proba:
            pred_proba = model.predict_proba(X)[:, 1]
            pred = (pred_proba > 0.5).astype(int)
            accuracy, f1, precision, recall, auc = eval_metrics(y, pred, pred_proba)
        else:
            pred = model.predict(X)
            accuracy, f1, precision, recall, auc = eval_metrics(y, pred)
        # Metrics
        # Logging best parameters from gridsearch if available
        if hasattr(model, 'best_params_'):
            mlflow.log_params(model.best_params_)
        # Log the metrics
        mlflow.log_metric("Mean CV score", model.best_score_ if hasattr(model, 'best_score_') else float('nan'))
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("AUC", auc)

        # Logging artifacts and model
        if use_proba:
            mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.sklearn.log_model(model, name)
        
        mlflow.end_run()

mlflow_logging(xgb_model, X_test, y_test, "XGBClassifier", use_proba=True)
mlflow_logging(svm_model, X_test, y_test, "SVM", use_proba=True)
