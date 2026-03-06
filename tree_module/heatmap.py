import numpy as np
import cv2

def generate_density_heatmap(image, predictions, grid_size=100):

    height, width, _ = image.shape

    heatmap = np.zeros((height//grid_size + 1, width//grid_size + 1))

    for _, row in predictions.iterrows():

        cx = int((row.xmin + row.xmax)/2)
        cy = int((row.ymin + row.ymax)/2)

        gx = cx // grid_size
        gy = cy // grid_size

        heatmap[gy, gx] += 1

    heatmap = cv2.resize(heatmap, (width, height))

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = heatmap.astype(np.uint8)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap