import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dataset_path = "dataset"
clases = os.listdir(dataset_path)

X = []
y = []

# cargar 500 imágenes por clase
for idx, clase in enumerate(clases):
    carpeta_clase = os.path.join(dataset_path, clase)
    imagenes = os.listdir(carpeta_clase)

    np.random.shuffle(imagenes)
    imagenes = imagenes[:500]

    for img_name in imagenes:
        img_path = os.path.join(carpeta_clase, img_name)

        img = Image.open(img_path).convert("L")
        img_array = np.array(img)

        img_vector = img_array.flatten()

        X.append(img_vector)
        y.append(idx)

X = np.array(X)
y = np.array(y)

print("Shape X:", X.shape)


# (a) DOS PRIMERAS COMPONENTES DEL VECTOR ORIGINAL

X_trunc = X[:, :2]

plt.figure()

for i in range(len(clases)):
    plt.scatter(
        X_trunc[y == i, 0],
        X_trunc[y == i, 1],
        label=clases[i],
        alpha=0.6
    )

plt.title("Vectores originales truncados a 2 componentes")
plt.xlabel("Pixel 1")
plt.ylabel("Pixel 2")
plt.legend()
plt.show()


# (b) PCA CON K = 2

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()

for i in range(len(clases)):
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        label=clases[i],
        alpha=0.6
    )

plt.title("Datos reducidos con PCA (K = 2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()