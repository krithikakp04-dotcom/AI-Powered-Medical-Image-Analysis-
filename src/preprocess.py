import cv2
import os
import numpy as np

IMG_SIZE = 224

def load_images(folder):
    data = []
    labels = []

    categories = ["NORMAL", "PNEUMONIA"]

    for category in categories:
        path = os.path.join(folder, category)
        label = categories.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append(img_array)
                labels.append(label)
            except:
                pass

    return np.array(data)/255.0, np.array(labels)