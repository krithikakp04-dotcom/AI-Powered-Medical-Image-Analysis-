import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -------------------------------
# STEP 1: CONFIG
# -------------------------------
IMG_SIZE = 224
dataset_path = "data/chest_xray/train"

data = []
labels = []
categories = ["NORMAL", "PNEUMONIA"]

print("🔄 Loading dataset...")

# -------------------------------
# STEP 2: LOAD DATASET
# -------------------------------
for category in categories:
    path = os.path.join(dataset_path, category)

    if not os.path.exists(path):
        print(f"❌ Path not found: {path}")
        continue

    label = categories.index(category)

    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)

            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img_array is None:
                continue

            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            data.append(img_array)
            labels.append(label)

        except:
            continue

print(f"✅ Loaded images: {len(data)}")

# -------------------------------
# STEP 3: PREPROCESS DATA
# -------------------------------
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# -------------------------------
# STEP 4: TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 5: BUILD MODEL
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# -------------------------------
# STEP 6: COMPILE MODEL
# -------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# STEP 7: TRAIN MODEL
# -------------------------------
print("🚀 Training started...")

history = model.fit(
    X_train, y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# -------------------------------
# STEP 8: SAVE MODEL
# -------------------------------
model.save("medical_model.h5")
print("🎉 Model Saved Successfully!")

# -------------------------------
# STEP 9: CREATE OUTPUT FOLDER
# -------------------------------
os.makedirs("outputs", exist_ok=True)

# -------------------------------
# STEP 10: ACCURACY GRAPH
# -------------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("outputs/accuracy.png")
plt.close()

# -------------------------------
# STEP 11: LOSS GRAPH
# -------------------------------
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("outputs/loss.png")
plt.close()

print("📊 Accuracy & Loss graphs saved!")

# -------------------------------
# STEP 12: CONFUSION MATRIX
# -------------------------------
print("📊 Generating Confusion Matrix...")

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).flatten()
y_true = y_test.astype(int)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png", bbox_inches='tight')
plt.close()

print("✅ Confusion Matrix saved successfully!")