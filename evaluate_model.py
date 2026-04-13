import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import os

# -------------------------------
# CREATE OUTPUT FOLDER
# -------------------------------
os.makedirs("outputs", exist_ok=True)

# -------------------------------
# PLOT TRAINING GRAPH
# -------------------------------
def plot_training(history):

    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("outputs/accuracy.png")
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/loss.png")
    plt.close()

    print("📊 Accuracy & Loss graphs saved!")


# -------------------------------
# MODEL EVALUATION
# -------------------------------
def evaluate_model(model, X_test, y_test):

    # Predictions
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("✅ Accuracy:", acc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    print("📊 Confusion matrix saved in outputs folder!")