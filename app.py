# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Step 2: Upload the file from file manager
from google.colab import files
uploaded = files.upload()

# Step 3: Read the uploaded CSV file
filename = list(uploaded.keys())[0]  # Automatically get uploaded file name
df = pd.read_csv(filename)

# Step 4: Split features and labels
X = df['question']
y = df['label']

# Step 5: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Vectorize text data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 7: Train logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 8: Evaluate the model
accuracy = model.score(X_test_vec, y_test)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Step 9: Save the model and vectorizer
joblib.dump(model, 'c_non_c_classifier.pkl')
joblib.dump(vectorizer, 'c_non_c_vectorizer.pkl')
print("🎉 Model and Vectorizer saved as 'c_non_c_classifier.pkl' and 'c_non_c_vectorizer.pkl'")
