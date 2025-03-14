import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.linear_model import LogisticRegression



# Download VADER Lexicon
nltk.download('vader_lexicon')

# Load dataset
file_path = r"D:\My Data\Mugdha jadhav\Desktop\oasis internship\sentiment analysis (task 2)\Twitter_Data.csv"  # Update with correct path
df = pd.read_csv(file_path)

# Convert 'clean_text' column to string and fill NaN values
df['clean_text'] = df['clean_text'].astype(str).fillna("")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Apply Sentiment Analysis on 'clean_text' column
df['sentiment_score'] = df['clean_text'].apply(lambda text: sia.polarity_scores(text)['compound'])

# Classify sentiment as Positive, Neutral, or Negative
df['sentiment_label'] = df['sentiment_score'].apply(lambda score: 'Positive' if score > 0.05 
                                                    else ('Negative' if score < -0.05 else 'Neutral'))

# Display results
print(df[['clean_text', 'sentiment_score', 'sentiment_label']].head())


# Set style
sns.set_style("whitegrid")

plt.figure(figsize=(8, 5))
sns.countplot(x="sentiment_label", data=df, hue="sentiment_label", palette={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}, legend=False)
plt.title("Sentiment Distribution", fontsize=14)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()



# Generate word clouds for each sentiment
for sentiment in ['Positive', 'Neutral', 'Negative']:
    text = " ".join(df[df['sentiment_label'] == sentiment]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Plot the word cloud
    plt.figure(figsize=(8, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Word Cloud for {sentiment} Sentiment", fontsize=14)
    plt.axis("off")
    plt.show()


# Convert Sentiment Labels into Numerical Classes
label_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['sentiment_class'] = df['sentiment_label'].map(label_mapping)

# Splitting data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment_class'], test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)  # Keep only 5000 most important words
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF Transformation Complete! ✅")

# Train Naïve Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Make Predictions
y_pred_nb = nb_model.predict(X_test_tfidf)

# Model Evaluation
print("Naïve Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))


lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

