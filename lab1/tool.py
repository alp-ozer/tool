########## 1. Import required libraries ##########
import pandas as pd
import numpy as np
import re

# Text preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

# Model training & evaluation
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc
)

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


########## 2. Define text preprocessing functions ##########

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# Stopwords
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list


def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])


def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


########## 3. Load and preprocess data ##########
# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'caffe'
path = f'datasets/{project}.csv'

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle

# Merge Title and Body into a single column; if Body is NaN, use Title only
pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

# Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

##########

# Load dataset
datafile = "Title+Body.csv"
data = pd.read_csv(datafile).fillna('')

# Define the text column
text_col = 'text'

# Apply text preprocessing
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

########## 4. Configure and train SVM ##########

REPEAT = 10  # Number of repeated experiments

# Lists to store evaluation metrics
accuracies, precisions, recalls, f1_scores, auc_values = [], [], [], [], []

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=1000)
X_all = tfidf.fit_transform(data[text_col])
X_all = X_all.toarray()  # Convert sparse matrix to NumPy array

for repeated_time in range(REPEAT):
    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(indices, test_size=0.2, random_state=repeated_time)

    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = data['sentiment'].iloc[train_index], data['sentiment'].iloc[test_index]

    svm_clf = SVC(kernel='linear', probability=True, random_state=repeated_time)
    svm_clf.fit(X_train, y_train)

    y_pred = svm_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precisions.append(prec)

    rec = recall_score(y_test, y_pred, average='macro')
    recalls.append(rec)

    f1 = f1_score(y_test, y_pred, average='macro')
    f1_scores.append(f1)

    fpr, tpr, _ = roc_curve(y_test, svm_clf.predict_proba(X_test)[:, 1])
    auc_val = auc(fpr, tpr)
    auc_values.append(auc_val)

########## 5. Aggregate results ##########

final_results = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
    "Value": [
        np.mean(accuracies),
        np.mean(precisions),
        np.mean(recalls),
        np.mean(f1_scores),
        np.mean(auc_values)
    ]
}

# Convert to DataFrame and display
results_df = pd.DataFrame(final_results)
print("=== SVM + TF-IDF Results ===")
print(results_df)
