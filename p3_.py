import pandas as pd
import ast
import joblib
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    hamming_loss,
    f1_score,
    accuracy_score,
    multilabel_confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

print("📥 Loading and validating data...")

def load_data(path):
    df = pd.read_csv(path)
    assert {'cleaned_summary', 'genres'}.issubset(df.columns)
    df['genres'] = df['genres'].apply(ast.literal_eval)
    return df

train_df = load_data("Dataset/train_cleaned.csv")
test_df = load_data("Dataset/test_cleaned.csv")

def clean_genres(genre_list):
    return [str(g).strip().lower() for g in genre_list if len(str(g).strip()) >= 4]

train_df['genres'] = train_df['genres'].apply(clean_genres)
test_df['genres'] = test_df['genres'].apply(clean_genres)

# Compute genre frequencies
all_genres = [g for genres in train_df['genres'] for g in genres]
genre_counts = pd.Series(all_genres).value_counts()
selected_genres = genre_counts[genre_counts >= 150].index.tolist()
print(f"🔍 Selected {len(selected_genres)} genres: {', '.join(selected_genres[:10])}...")

# Filter rows that have at least one of the selected genres
def filter_by_selected(df, allowed_genres):
    return df[df['genres'].apply(lambda gs: any(g in allowed_genres for g in gs))].copy()

train_df = filter_by_selected(train_df, selected_genres)
test_df = filter_by_selected(test_df, selected_genres)

# Binarize targets using only selected genres
mlb = MultiLabelBinarizer(classes=selected_genres)
y_train = mlb.fit_transform(train_df['genres'])
y_test = mlb.transform(test_df['genres'])

# Define text vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=3,
    max_df=0.75,
    sublinear_tf=True
)

# Define classifier
clf = MultiOutputClassifier(
    LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='liblinear'
    )
)

# Full pipeline
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('select', SelectKBest(chi2, k=3000)),
    ('clf', clf)
])

print("🤖 Training optimized model...")
pipeline.fit(train_df['cleaned_summary'], y_train)

# Evaluate
print("📊 Evaluating model performance...")
y_pred = pipeline.predict(test_df['cleaned_summary'])

print("\n🔥 Key Metrics:")
print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.4f}")
print(f"Micro F1 Score: {f1_score(y_test, y_pred, average='micro'):.4f}")
print(f"Subset Accuracy (Exact Match Ratio): {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=selected_genres, zero_division=0))

# Confusion Matrices for selected genres - Modified to show all top 5 in one figure
print("\n📉 Plotting Confusion Matrices for top 5 genres...")
conf_matrices = multilabel_confusion_matrix(y_test, y_pred)

# Create a single figure with subplots for all 5 genres
plt.figure(figsize=(15, 10))
for idx, genre in enumerate(selected_genres[:5]):  # Plot top 5 genres
    cm = conf_matrices[idx]
    plt.subplot(2, 3, idx+1)  # 2 rows, 3 columns layout (5 plots)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'"{genre.title()}"', fontsize=10)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
plt.suptitle('Confusion Matrices for Top 5 Genres', fontsize=12, y=1.02)
plt.tight_layout()
plt.show()

# Save model and label binarizer
print("💾 Saving models...")
joblib.dump(pipeline, "models/genre_model.pkl", compress=3)
joblib.dump(mlb, "models/mlb.pkl", compress=3)
joblib.dump(selected_genres, "models/selected_genres.pkl", compress=3)

print("✅ Done! Use these files in your GUI.")