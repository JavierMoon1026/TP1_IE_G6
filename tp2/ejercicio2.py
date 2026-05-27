import cv2
import numpy as np
import matplotlib.pyplot as plt
import ejercicio1

# Funcion para cargar imagenes
def cargar_imagen(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("No se pudo cargar la imagen")

    return img


# Imagen de referencia
img_ref_path = "tp2/Imagenes/img01.jpg"

# Imagenes a comparar
imagenes = {
    "Desplazada 1": "tp2/Imagenes/img01_desplazada1.jpg",
    "Desplazada 2": "tp2/Imagenes/img01_desplazada2.jpg",

    "Zoom 1": "tp2/Imagenes/img01_zoom1.jpg",
    "Zoom 2": "tp2/Imagenes/img01_zoom2.jpg",

    "Contraste 1": "tp2/Imagenes/img01_contraste1.jpg",
    "Contraste 2": "tp2/Imagenes/img01_contraste2.jpg"
}


# Calculo y graficos
for titulo, path in imagenes.items():

    # Calcular informacion mutua
    mi = ejercicio1.mutual_information(img_ref_path, path)

    print(f"{titulo}: IM = {mi:.4f}")

    # Cargar imagenes para mostrar
    img_ref = cv2.imread(img_ref_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Igualar tamanos
    h = min(img_ref.shape[0], img.shape[0])
    w = min(img_ref.shape[1], img.shape[1])

    img_ref = img_ref[:h, :w]
    img = img[:h, :w]

    # Graficos
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img_ref, cmap="gray")
    ax[0].set_title("Imagen referencia")
    ax[0].axis("off")

    ax[1].imshow(img, cmap="gray")
    ax[1].set_title(f"{titulo}\nMI = {mi:.4f}")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()