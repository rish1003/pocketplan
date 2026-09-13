from sklearn.svm import LinearSVC

svm_model = LinearSVC()

svm_model.fit(X_train_vec, y_train)

y_pred = svm_model.predict(X_test_vec)

from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)