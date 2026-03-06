import cv2
import numpy as np

def compute_ndvi_proxy(image):

    B,G,R = cv2.split(image.astype(float))

    ndvi = (G - R) / (G + R + 1e-6)

    ndvi = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX)

    ndvi = ndvi.astype(np.uint8)

    ndvi_map = cv2.applyColorMap(ndvi, cv2.COLORMAP_SUMMER)

    return ndvi_map, ndvi