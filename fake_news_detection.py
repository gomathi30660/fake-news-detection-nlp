import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the datasets
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

print("Fake news shape:", fake_df.shape)
print("Real news shape:", real_df.shape)

# Step 2: Add labels
fake_df["label"] = 0  # Fake news
real_df["label"] = 1  # Real news

# Step 3: Combine datasets
combined_df = pd.concat([fake_df, real_df], axis=0).reset_index(drop=True)

# Step 4: Prepare a clean dataframe with only title, text, label
clean_df = combined_df[["title", "text", "label"]].copy()
clean_df["content"] = clean_df["title"] + " " + clean_df["text"]
clean_df = clean_df[["content", "label"]]

# Step 5: Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

clean_df["content"] = clean_df["content"].apply(clean_text)

print("\nCleaned sample:")
print(clean_df["content"].head())

# Step 6: TF-IDF Vectorization
X = clean_df["content"]
y = clean_df["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

print("\nVectorization complete!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Step 8: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 9: Predict and evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
import pickle

# Save the trained model
with open("fake_news_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)


