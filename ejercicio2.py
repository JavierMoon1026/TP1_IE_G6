import os
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

dataset_path = "dataset"
clases = os.listdir(dataset_path)

X = []
y = []

for idx, clase in enumerate(clases):
    carpeta_clase = os.path.join(dataset_path, clase)
    imagenes = os.listdir(carpeta_clase)
    # mezclar aleatoriamente
    np.random.shuffle(imagenes)
    # quedarte con 500
    imagenes = imagenes[:500]

    for img_name in imagenes:
        img_path = os.path.join(carpeta_clase, img_name)
        # escala de grises
        img = Image.open(img_path).convert("L")
        # convertir a array
        img_array = np.array(img)
        # aplanar
        img_vector = img_array.flatten()
        
        X.append(img_vector)
        y.append(idx)

X = np.array(X)
y = np.array(y)

print("Shape X:", X.shape)
print("Shape y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

#2a)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc_sin_pca = accuracy_score(y_test, y_pred)

print("Accuracy sin PCA:", acc_sin_pca)

#2b)