import os
import subprocess

# Define the dataset name and destination path
dataset = 'chethuhn/network-intrusion-dataset'
destination_path = '/tmp/network-intrusion-dataset'

# Check if the dataset already exists
if not os.path.exists(destination_path):
    print(f"Dataset not found at {destination_path}\nDownloading...")
    os.makedirs(destination_path, exist_ok=True)  # Ensure the destination directory exists
    
    # Keep trying to download until it succeeds, in case of network connection issues.
    while True:
        try:
            subprocess.run(f"kaggle datasets download -d {dataset} -p {destination_path} --unzip", 
                           shell=True, 
                           check=True,
                           stderr=subprocess.STDOUT)
            print("Download complete.")
            break  # Exit the loop when download succeeds
        except subprocess.CalledProcessError:
            print(f"Download failed.\nRetrying...")
else:
    print(f"Dataset already exists at {destination_path}\nSkipping download.")
import pandas as pd
import glob

# Get all CSV files in the specified directory
csv_files = glob.glob(os.path.join(destination_path, "*.csv"))

# Order for the days of the week based on the dataset documentation
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Sort the files based on day and time of day (morning before afternoon)
csv_files_sorted = sorted(csv_files, key=lambda x: (
    days_order.index(next(day for day in days_order if day in os.path.basename(x))),
    'Afternoon' in os.path.basename(x)
))

# Combine all CSVs into one DataFrame in the sorted order
df = pd.concat([pd.read_csv(file) for file in csv_files_sorted], ignore_index=True)

print("Data loaded successfully.\nShape:", df.shape)
print("Data Columns:", df.columns.tolist())
df.sample(n=10, random_state=42)
import numpy as np

# Check for missing values
print(f"Total missing values in the DataFrame: {df.isnull().sum().sum()}\n")
print(f"Number of missing values in the first 20 columns:\n{df.isnull().sum().head(20)}\n")

# Remove spaces at the beginning and at the end of the column names
df.columns = df.columns.str.strip()

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Use median imputation
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())
    
# Check for missing values after cleaning
print(f"Total missing values after cleaning: {df.isnull().sum().sum()}\n")

print("Data Columns:", df.columns.tolist())
print(df['Label'].value_counts().sum)
df['Label'] = np.where(df['Label'].isin(['BENIGN']), 0, 1)
print(df['Label'].value_counts())
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style to dark background with light text and lines
plt.style.use('dark_background')

sns.countplot(x='Label', data=df, palette=['b', 'r'], hue='Label', legend=False)

plt.show()
X = df.drop('Label', axis=1)
y = df['Label']
X.sample(5, random_state=42)
y.sample(5, random_state=42)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data, and transform the test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.decomposition import PCA

# Define the desired explained variance threshold (e.g., 95%)
explained_variance_threshold = 0.95

# Fit PCA without specifying the number of components to compute all components
pca = PCA()
pca.fit(X_train)

# Calculate the cumulative explained variance
cumulative_variance = pca.explained_variance_ratio_.cumsum()

# Find the number of components that meet the threshold
n_components_best = (cumulative_variance >= explained_variance_threshold).argmax() + 1

print(f"Original features count: {X_train.shape[1]}")

# Apply PCA with the optimal number of components
pca = PCA(n_components=n_components_best)
X_train = pca.fit_transform(X_train)

# Transform the test data using the same PCA model
X_test = pca.transform(X_test)

print(f"Reduced features count: {X_train.shape[1]}")
import matplotlib.pyplot as plt

# Set the style to dark background with light text and lines
plt.style.use('dark_background')

plt.figure(figsize=(10, 4))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, color='b', marker='*', linestyle='dotted')
plt.axhline(y=explained_variance_threshold, color='r', linestyle='-')
plt.axvline(x=n_components_best, color='g', linestyle='-')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()
def create_sequences(X, y, sequence_length):
    sequences_X = [X[i:i + sequence_length] for i in range(0, len(X) - sequence_length + 1, sequence_length)]
    # The corresponding labels for each sequence (using the last label in the sequence)
    sequences_y = [y.iloc[i + sequence_length - 1] for i in range(0, len(X) - sequence_length + 1, sequence_length)]
    return np.array(sequences_X), np.array(sequences_y)

sequence_length = 100

# Create sequential data for training and testing
X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

print("\nData shapes after processing:")
print(f"X_train_seq shape: {X_train_seq.shape}\nX_test_seq shape: {X_test_seq.shape}")
print(f"y_train_seq shape: {y_train_seq.shape}\ny_test_seq shape: {y_test_seq.shape}")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
import keras_tuner as kt
from keras_tuner import HyperModel

class LSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        
        model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))))
        model.add(Bidirectional(LSTM(64, return_sequences=False, activation='tanh')))
        model.add(Dropout(0.2)) # Dropout layer for regularization
        model.add(Dense(64))  # 不指定激活函数
        model.add(tf.keras.layers.LeakyReLU())  # 单独添加 LeakyReLU 激活层
        model.add(Dense(1, activation='sigmoid')) # Single output for binary classification
        
        model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

# Initialize Keras Tuner with RandomSearch
tuner = kt.RandomSearch(LSTMHyperModel(), objective='val_accuracy', max_trials=5, executions_per_trial=2)

# Implement EarlyStopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Search for the best hyperparameters
tuner.search(X_train_seq, y_train_seq, epochs=10, validation_data=(X_test_seq, y_test_seq), callbacks=[early_stopping])

# Get the best model
best_model = tuner.get_best_models()[0]

# Train the best model
history = best_model.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq), epochs= 30, batch_size=32, callbacks=[early_stopping])
from sklearn.preprocessing import Binarizer

# Initialize the Binarizer with a custom threshold (default is 0.0)
binarizer = Binarizer(threshold=0.5)

y_pred_prob = best_model.predict(X_test_seq)

# Apply binarizer to the 
y_pred = binarizer.fit_transform(y_pred_prob.reshape(-1, 1)).flatten()
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(y_test_seq, y_pred)
print(f"Accuracy: {accuracy:.4f}")

recall = recall_score(y_test_seq, y_pred)
print(f"Recall: {recall:.4f}")

f1 = f1_score(y_test_seq, y_pred)
print(f"F1 Score: {f1:.4f}")

roc_auc = roc_auc_score(y_test_seq, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")
from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test_seq, y_pred))
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_seq, y_pred_prob)
roc_auc_val = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_seq, y_pred)
f, ax = plt.subplots(figsize=(5, 5))

sns.heatmap(cm, annot=True, fmt='.0f', linewidths=0.5, linecolor="r", ax=ax)

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()