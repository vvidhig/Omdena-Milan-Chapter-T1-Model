import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import mlflow
import mlflow.tensorflow
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout

import warnings
warnings.filterwarnings("ignore")

# Set tracking URI and experiment
# mlflow.set_tracking_uri("http://192.168.0.1:5000")
mlflow.set_experiment("Urban_Farming_Prediction_BiLSTM")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)

# Normalize the feature values
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data to be suitable for LSTM (samples, time steps, features)
X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define the BiLSTM model
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, X_train_scaled.shape[2])))
model.add(Dropout(0.3))  # Adding Dropout layer to prevent overfitting
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)

# Make predictions and evaluate the model
y_pred_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

recall = metrics.recall_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

print(f'BiLSTM Model - Accuracy: {accuracy}, Recall: {recall}, AUC: {auc}, Precision: {precision}')

# Model evaluation metrics function
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
        plt.savefig("plots/ROC_curve_BiLSTM.png")
        # Close plot
        plt.close()
    else:
        auc = float('nan')
    return accuracy, f1, precision, recall, auc

# MLflow logging function
def mlflow_logging(model, X, y, name, use_proba=False):
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        if use_proba:
            pred_proba = model.predict(X).flatten()
            pred = (pred_proba > 0.5).astype(int)
            accuracy, f1, precision, recall, auc = eval_metrics(y, pred, pred_proba)
        else:
            pred = model.predict(X).flatten()
            accuracy, f1, precision, recall, auc = eval_metrics(y, (pred > 0.5).astype(int))
        # Metrics
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("AUC", auc)

        # Logging artifacts and model
        if use_proba:
            mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.tensorflow.log_model(model, name)
        
        mlflow.end_run()

mlflow_logging(model, X_test_scaled, y_test, "BiLSTM", use_proba=True)
