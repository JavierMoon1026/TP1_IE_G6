import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

np.random.seed(6)

ruta_base = 'dataset'

nombres_clases = sorted(
    [d for d in os.listdir(ruta_base) if os.path.isdir(os.path.join(ruta_base, d))]
)

X = []
y = []

for etiqueta_numerica, nombre_clase in enumerate(nombres_clases):
    ruta_carpeta = os.path.join(ruta_base, nombre_clase)
    rutas_imagenes = glob.glob(os.path.join(ruta_carpeta, "*.png"))

    rutas_seleccionadas = np.random.choice(rutas_imagenes, size=500, replace=False)

    for ruta_img in rutas_seleccionadas:
        img = Image.open(ruta_img).convert('L')
        X.append(np.array(img).flatten())
        y.append(etiqueta_numerica)

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
k_values = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 784]
acc_con_pca = []

for k in k_values:
    k_real = min(k, X_train.shape[0], X_train.shape[1])

    pca = PCA(n_components=k_real)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)

    y_pred = knn.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)

    acc_con_pca.append(acc)

    print(f"K = {k_real:3d} --> Accuracy = {acc:.4f}")

#grafico
plt.figure(figsize=(8, 5))
plt.plot(k_values, acc_con_pca, marker="o", label="K-NN con PCA")
plt.axhline(acc_sin_pca, color="red", linestyle="--", label="K-NN sin PCA")

plt.xscale("log")
plt.xlabel("Cantidad de componentes principales K")
plt.ylabel("Accuracy")
plt.title("Accuracy en función de K")
plt.grid(True)
plt.legend()
plt.show()