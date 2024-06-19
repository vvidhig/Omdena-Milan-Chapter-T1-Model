import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import optuna
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
file_path = 'dataset/Merged_2014.csv'
data = pd.read_csv(file_path)

# Select categorical columns
categorical_cols = ['Zone', 'landuse']

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical_cols = encoder.fit_transform(data[categorical_cols])

# Create a DataFrame for the encoded features
encoded_df = pd.DataFrame(encoded_categorical_cols, columns=encoder.get_feature_names_out(categorical_cols))

# Drop original categorical columns and concatenate the encoded columns
data_encoded = data.drop(columns=categorical_cols).join(encoded_df)

# Handle NaN values by filling them with the mean of the respective columns
data_encoded.fillna(data_encoded.mean(), inplace=True)

# Separate features and labels
X_encoded = data_encoded.drop(columns=['Suitable_Areas'])
y_encoded = data_encoded['Suitable_Areas']

# Normalize the feature values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Hyperparameter Optimization with Optuna for XGBClassifier
def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }

    model = XGBClassifier(**param)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# Train XGBClassifier and Evaluate
best_params = study.best_trial.params
xgb_model = XGBClassifier(**best_params)
xgb_model.fit(X_train, y_train)

xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred = (xgb_pred_proba > 0.5).astype(int)

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_pred_proba)

print(f'XGBClassifier - Accuracy: {xgb_accuracy}, Recall: {xgb_recall}, AUC: {xgb_auc}')

# Train SVM and Evaluate
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

svm_pred_proba = svm_model.predict_proba(X_test)[:, 1]
svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_auc = roc_auc_score(y_test, svm_pred_proba)

print(f'SVM - Accuracy: {svm_accuracy}, Recall: {svm_recall}, AUC: {svm_auc}')
