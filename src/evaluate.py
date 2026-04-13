# evaluate.py

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))