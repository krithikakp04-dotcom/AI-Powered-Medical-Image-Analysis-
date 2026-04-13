from preprocess import load_images
from model import build_model
from sklearn.model_selection import train_test_split

X, y = load_images("data/train")

X = X.reshape(-1, 224, 224, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = build_model()

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save("models/medical_model.h5")