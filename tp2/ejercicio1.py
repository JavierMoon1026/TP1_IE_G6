import numpy as np
import cv2

carpeta_imagenes ="Imagenes"

def mutual_information(Img1,Img2):
    """
    Img1 e Img2 son las imagenes
    
    """
    img1= np.array(cv2.imread(Img1,cv2.IMREAD_GRAYSCALE))
    img2= np.array(cv2.imread(Img2,cv2.IMREAD_GRAYSCALE))

    vector1 = img1.flatten()
    vector2 = img2.flatten()

    hist_2d, _, _ = np.histogram2d(vector1, vector2, bins=256)

    p_xy = hist_2d/ len(vector2)
    p_x = np.sum(p_xy,axis=1)
    p_y = np.sum(p_xy,axis=0)
    p_x_p_y = np.outer(p_x,p_y)

    nonzero_mask = p_xy > 0
    
    mi = np.sum(p_xy[nonzero_mask] * np.log2(p_xy[nonzero_mask] / p_x_p_y[nonzero_mask]))
    
    return mi