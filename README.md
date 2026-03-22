# Sentiment-Classification of Pride and Prejudice Comments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import re # import the regular expressions module
from sklearn.model_selection import train_test_split # Import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

%matplotlib inline
sns.set_style('whitegrid')
pd.set_option('display.max_colwidth', 150)

df = pd.read_excel('Pride and Prejudice comments.xls')
df.rename(columns={'positive': 'n_pos_words', 'negative': 'n_neg_words'}, inplace=True)

df_binary = df[df['sentiment'].isin(['pos', 'neg'])].copy()

print("\nDataframe dimensions (rows, columns) after filtering for 'pos' and 'neg':")
print(df_binary.shape)

print("\nColumn labels:\n")
for col in df_binary.columns:
    print(col)

print("\nNumber of positive and negative reviews (after filtering):")
print(df_binary['sentiment'].value_counts())

X = df_binary[['n_pos_words', 'n_neg_words']]
y = df_binary['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # Removed stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print("Train set sentiment distribution:")
print(y_train.value_counts(normalize=True))
print("Test set sentiment distribution:")
print(y_test.value_counts(normalize=True))

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

pos_class_idx = np.where(model.classes_ == 'pos')[0][0]
y_pred_proba = model.predict_proba(X_test)[:, pos_class_idx]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='pos', zero_division=0)
recall = recall_score(y_test, y_pred, pos_label='pos', zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label='pos', zero_division=0)

print("=" * 50)
print("result")
print("=" * 50)
print(f" (Accuracy):  {accuracy:.4f} ({accuracy:.2%})")
print(f" (Precision): {precision:.4f} ({precision:.2%})")
print(f" (Recall):    {recall:.4f} ({recall:.2%})")
print(f" (F1-Score):  {f1:.4f} ({f1:.2%})")
print("=" * 50)

print("\nreport：")
print(classification_report(y_test, y_pred, target_names=['neg', 'pos'], zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=['neg', 'pos']) # Specify labels to ensure order

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['neg', 'pos'],
            yticklabels=['neg', 'pos'])
plt.title('confusion matrix', fontsize=14)
plt.xlabel('prediction_label')
plt.ylabel('true_label')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label='pos')
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('(False Positive Rate)')
plt.ylabel('(True Positive Rate)')
plt.title('ROC')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
