from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

word_vec = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=5000
)

X_train_vec = hstack([
    word_vec.fit_transform(X_train),
    char_vec.fit_transform(X_train)
])

X_test_vec = hstack([
    word_vec.transform(X_test),
    char_vec.transform(X_test)
])

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    max_iter=2000,
    solver="saga",
    C=1,
    n_jobs=-1
)

clf.fit(X_train_vec, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = clf.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))