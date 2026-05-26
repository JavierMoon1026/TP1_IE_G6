import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


ruta_base = 'dataset'

nombres_clases = [d for d in os.listdir(ruta_base)]

X_lista = []
y_lista = []

for etiqueta_numerica, nombre_clase in enumerate(nombres_clases):
    ruta_carpeta = os.path.join(ruta_base, nombre_clase)
    rutas_imagenes = glob.glob(os.path.join(ruta_carpeta, "*.png")) 
    
    rutas_seleccionadas = np.random.choice(rutas_imagenes, size=500, replace=False)
    
    for ruta_img in rutas_seleccionadas:
        img = Image.open(ruta_img).convert('L')
        X_lista.append(np.array(img).flatten())
        y_lista.append(etiqueta_numerica)

X = np.array(X_lista)
y = np.array(y_lista)
clases = np.unique(y)

k_values = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 784]

var_exp_dict = {c: [] for c in clases}
mse_dict = {c: [] for c in clases}

for c in clases:
    X_c = X[y == c]
    
    for k in k_values:
        k_real = min(k, len(X_c)) 
        
        pca = PCA(n_components=k_real)
        X_c_red = pca.fit_transform(X_c)
        X_c_rec = pca.inverse_transform(X_c_red)
        
        var_exp_dict[c].append(np.sum(pca.explained_variance_ratio_))
        mse_dict[c].append(mean_squared_error(X_c, X_c_rec))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

for c, nombre in zip(clases, nombres_clases):
    ax[0].plot(k_values, var_exp_dict[c], label=nombre)
ax[0].set_title('Varianza Explicada vs. K')
ax[0].set_xlabel('K')
ax[0].set_ylabel('Varianza Explicada')
ax[0].set_xscale('log')
ax[0].grid(True)
ax[0].legend()

# Gráfico de MSE
for c, nombre in zip(clases, nombres_clases):
    ax[1].plot(k_values, mse_dict[c], label=nombre)
ax[1].set_title('MSE vs. K')
ax[1].set_xlabel('K (Componentes Principales)')
ax[1].set_ylabel('MSE')
ax[1].set_xscale('log')
ax[1].grid(True)
ax[1].legend()

plt.show()

fig, axes = plt.subplots(nrows=len(clases), ncols=3, figsize=(10, 12))

for i, c in enumerate(clases):
    X_c = X[y == c]

    img_orig = X_c[0]

    pca_50 = PCA(n_components=50).fit(X_c)
    img_rec_50 = pca_50.inverse_transform(pca_50.transform([img_orig]))[0]

    pca_2 = PCA(n_components=2).fit(X_c)
    img_rec_2 = pca_2.inverse_transform(pca_2.transform([img_orig]))[0]

    axes[i, 0].imshow(img_orig.reshape(28, 28), cmap='gray')
    axes[i, 0].set_title(f'Original - {nombres_clases[i]}')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(img_rec_50.reshape(28, 28), cmap='gray')
    axes[i, 1].set_title('Reconstruida K=50')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(img_rec_2.reshape(28, 28), cmap='gray')
    axes[i, 2].set_title('Reconstruida K=2')
    axes[i, 2].axis('off')


plt.show()